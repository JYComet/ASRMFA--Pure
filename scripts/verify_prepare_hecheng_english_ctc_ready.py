#!/usr/bin/env python3
"""Tempdir-only checks for the strict English CTC-ready preparer."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import prepare_hecheng_english_ctc_ready as prep
import ctc_prealign as ctc
import normalize_english_tokens
from pipeline_utils import compute_model_tree_digest, write_ctc_run_receipt
from postprocess_textgrids import Interval, TextGrid, Tier, parse_textgrid, write_textgrid

def ok(value, label): print(("OK " if value else "FAIL ") + label); return not value
def wav(path):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000); handle.writeframes(b"\0\0" * 16000)
def bundle(root, stem, valid=True, token_end=1.):
    root.mkdir(parents=True, exist_ok=True); end = 1. if valid else 0.
    write_textgrid(TextGrid(0, 1, [Tier("words", 0, 1, [Interval(0, end, "ni3")]), Tier("pauses", 0, 1, [Interval(.2, .3, "sp")])]), root / f"{stem}.TextGrid")
    (root / f"{stem}.lab").write_text("ni3\n"); (root / f"{stem}_tokens.jsonl").write_text(json.dumps({"word":"ni3","start_s":0,"end_s":token_end,"start_ms":0,"end_ms":token_end*1000})+"\n")
    (root / f"{stem}_punct.json").write_text("[]\n"); (root / f"{stem}_text_cn.txt").write_text("你\n"); (root / f"{stem}_text_raw.txt").write_text("你\n"); (root / f"{stem}_ref.txt").write_text("ref\n")
def fixture(root):
    source, legacy, dictionary = root/"source", root/"legacy", root/"dict"
    source.mkdir(); legacy.mkdir(); dictionary.write_text("ni3 n i3\n")
    for stem in ("good1","good2","bad","noref"): wav(source/f"{stem}.wav")
    for stem in ("good1","good2","bad"): (source/f"{stem}.txt").write_text("ref\n")
    # A normal TextGrid can use a different word end than the canonical CTC token end.
    bundle(legacy,"good1", token_end=.9); bundle(legacy,"good2"); bundle(legacy,"bad",False)
    return source, legacy, dictionary
def args(source, legacy, dictionary, run):
    # The v4 preparer fingerprints the model tree before any run-root write.
    # Keep this fixture hermetic while exercising that production gate.
    model = run.parent / "model"
    model.mkdir(exist_ok=True)
    return SimpleNamespace(run_root=run,source_dir=source,legacy_ctc=legacy,rerun_ctc=None,dictionary_source=dictionary,
      require_expected_counts=True,expected_wavs=4,expected_txts=3,expected_authoritative=3,expected_missing_refs=1,expected_txt_only=0,expected_standard=2,expected_canonicalize=0,expected_rerun=1,expected_missing_stems=["noref"],asr_python="python",asr_model=str(model),asr_device="cuda:7")
def malformed_textgrid(path, *, bad_tail=False, token_end=.999, word_index=0, word_domain=False):
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', '', 'xmin = 0', 'xmax = 1', 'tiers? <exists>', 'size = 2', 'item []:', '    item [1]:', '        class = "IntervalTier"', '        name = "words"']
    if word_domain: lines += ['        xmin = 0', '        xmax = 1']
    lines += ['        intervals: size = 1', '    item [2]:', '        class = "IntervalTier"', f'        intervals [{word_index}]:', '            xmin = 0', f'            xmax = {token_end}', '            text = "ni3"', '        name = "pauses"', '        xmin = 0', '        xmax = 1', '        intervals: size = 1', '        intervals [1]:', '            xmin = 0.2', '            xmax = 0.3', '            text = "sp"']
    if bad_tail: lines.append('    unexpected = 1')
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
def main():
  fails=0
  with tempfile.TemporaryDirectory() as td:
    root=Path(td); source,legacy,dictionary=fixture(root); report=prep.inspect(source,legacy); a=args(source,legacy,dictionary,root/"run")
    fails+=ok(report["legacy_valid"]==["good1","good2"] and report["needs_rerun"]==["bad"] and report["missing_reference"]==["noref"],"classifies 2 valid, invalid, missing reference")
    old=root/"old"; old.mkdir(); bundle(old,"known"); malformed_textgrid(old/"known.TextGrid")
    category, issues, parsed=prep.classify_ctc_bundle(old,"known",source/"good1.wav")
    converted=root/"converted.TextGrid"; before=(old/"known.TextGrid").read_bytes()
    transform=prep.canonicalize_legacy_textgrid(old/"known.TextGrid",converted,parsed,root,stem="known",token_path=old/"known_tokens.jsonl",wav_path=source/"good1.wav") if category=="canonicalize" else {}
    canonical_words=[iv for iv in parse_textgrid(converted).tiers if iv.name=="words"][0].intervals
    fails+=ok(category=="canonicalize" and not issues and before==(old/"known.TextGrid").read_bytes() and prep._standard_textgrid(converted,prep.load_ctc_token_entries(old/"known_tokens.jsonl"),source/"good1.wav") and canonical_words[0].xmax==.999 and transform["word_count"]==1 and transform["pause_count"]==1 and transform["parser_signature"]==prep.LEGACY_PARSER_SIGNATURE and transform["transformation_version"]==prep.TRANSFORMATION_VERSION and transform["token_interval_mapping"]==[{"token_index":0,"canonical_word_ordinal":0,"textgrid_interval_index":1}],"observed zero-based/no-domain grammar canonicalizes with exact legacy end and 1-based mapping")
    malformed_textgrid(old/"known.TextGrid",bad_tail=True)
    fails+=ok(prep.classify_ctc_bundle(old,"known",source/"good1.wav")[0] is None,"unknown malformed grammar is rerun-only")
    malformed_textgrid(old/"known.TextGrid",word_index=1)
    one_based_rejected=prep.classify_ctc_bundle(old,"known",source/"good1.wav")[0] is None
    malformed_textgrid(old/"known.TextGrid",word_domain=True)
    domain_rejected=prep.classify_ctc_bundle(old,"known",source/"good1.wav")[0] is None
    fails+=ok(one_based_rejected and domain_rejected,"one-based or words-domain malformed variants are rerun-only")
    malformed_textgrid(old/"known.TextGrid")
    _, _, parsed=prep.classify_ctc_bundle(old,"known",source/"good1.wav"); transformed=root/"ctc_ready"/"known.TextGrid"
    transform=prep.canonicalize_legacy_textgrid(old/"known.TextGrid",transformed,parsed,root,stem="known",token_path=old/"known_tokens.jsonl",wav_path=source/"good1.wav")
    transform_report={"legacy_ctc":str(old),"legacy_canonicalize":["known"]}; transform_args=SimpleNamespace(run_root=root)
    prep.verify_transforms([transform],transform_args,transform_report)
    transform["token_interval_mapping"][0]["textgrid_interval_index"]=9
    try: prep.verify_transforms([transform],transform_args,transform_report); transform_mapping=False
    except ValueError: transform_mapping=True
    transform["token_interval_mapping"][0]["textgrid_interval_index"]=1
    transform["parser_signature"]="wrong"
    try: prep.verify_transforms([transform],transform_args,transform_report); transform_signature=False
    except ValueError: transform_signature=True
    transform["parser_signature"]=prep.LEGACY_PARSER_SIGNATURE; transform["word_count"]=9
    try: prep.verify_transforms([transform],transform_args,transform_report); transform_counts=False
    except ValueError: transform_counts=True
    transform["word_count"]=1; transform["transformation_version"]="wrong"
    try: prep.verify_transforms([transform],transform_args,transform_report); transform_version=False
    except ValueError: transform_version=True
    transform["transformation_version"]=prep.TRANSFORMATION_VERSION
    old_token=(old/"known_tokens.jsonl").read_text(); (old/"known_tokens.jsonl").write_text(old_token.replace('"ni3"','"bad"'))
    try: prep.verify_transforms([transform],transform_args,transform_report); transform_source=False
    except ValueError: transform_source=True
    (old/"known_tokens.jsonl").write_text(old_token)
    fails+=ok(transform_mapping and transform_signature and transform_counts and transform_version and transform_source,"canonical transform mapping, parser/count/version, and source-token tamper are rejected")
    rewrite=root/"rewrite.TextGrid"
    write_textgrid(TextGrid(0,1,[Tier("words",0,1,[Interval(0,.2,"ru i"),Interval(.2,1,"ya4")]),Tier("pauses",0,1,[Interval(.3,.4,"sp")]),Tier("other",0,1,[Interval(0,1,"keep")])]),rewrite)
    new_tokens=[{"word":"ria\"x","start_s":.1,"end_s":.8}]
    before_pause=[(.3,.4,"sp")]; normalize_english_tokens.rewrite_ctc_textgrid_words(rewrite,new_tokens); rewritten=parse_textgrid(rewrite)
    rw_words=[x for x in rewritten.tiers if x.name=="words"][0].intervals; rw_pauses=[x for x in rewritten.tiers if x.name=="pauses"][0].intervals
    fails+=ok([x.text for x in rw_words]==["",'ria"x',""] and [(x.xmin,x.xmax,x.text) for x in rw_pauses]==before_pause and 'intervals [1]:' in rewrite.read_text() and 'intervals [0]:' not in rewrite.read_text(),"normalizer rewrites standard tier atomically with blanks, escaping, 1-based blocks and preserved pauses")
    try: normalize_english_tokens.rewrite_ctc_textgrid_words(old/"known.TextGrid",new_tokens); malformed_rewrite=False
    except ValueError: malformed_rewrite=True
    fails+=ok(malformed_rewrite,"normalizer rejects old malformed grammar")
    extra=root/"extra.TextGrid"
    write_textgrid(TextGrid(0,1,[Tier("words",0,1,[Interval(0,1,"ni3")]),Tier("pauses",0,1,[Interval(.2,.3,"sp")]),Tier("extra",0,1,[Interval(0,1,"x")])]),extra)
    try: prep._standard_textgrid(extra,prep.load_ctc_token_entries(legacy/"good2_tokens.jsonl"),source/"good1.wav"); extra_tier=False
    except ValueError: extra_tier=True
    fails+=ok(extra_tier,"standard validator rejects extra/unknown tiers")
    missing=root/"missing"; missing.mkdir(); bundle(missing,"x"); (missing/"x.lab").unlink()
    missing_result=prep.classify_ctc_bundle(missing,"x",source/"good1.wav")
    fails+=ok(missing_result[0] is None and isinstance(missing_result[1],list) and missing_result[2] is None,"missing suffix returns fail-closed classification tuple")
    fails+=ok(not a.run_root.exists(),"inspect writes nothing")
    command=prep.render_rerun_command(a); fails+=ok("--pinyin-dir" in command and "--stems-file" not in command and "--all-gpus" not in command and "--device" in command and "cuda:7" in command and command.count("--no-nvv") == 1,"render command uses supported reference-only CLI")
    help_result=subprocess.run([sys.executable,str(PROJECT_ROOT/"scripts"/"ctc_prealign.py"),"--help"],text=True,capture_output=True)
    fails+=ok(help_result.returncode==0 and "--no-dict-update" in help_result.stdout and "--no-dict-update" in command,"CTC help and rendered command expose no-dict-update")
    ctc_out=root/"ctc-no-dict"; ctc_out.mkdir(); (ctc_out/"x.lab").write_text("newenglish\n"); (ctc_out/"x_text_cn.txt").write_text("x\n")
    ctc_dict=root/"ctc-no-dict.dict"; ctc_dict.write_text("ni3 n i3\n"); before_ctc=prep.sha256(ctc_dict)
    with patch.object(normalize_english_tokens,"normalize_stem",return_value=True):
      ctc._normalize_english(ctc_out,ctc_dict,update_dict=False)
    fails+=ok(prep.sha256(ctc_dict)==before_ctc,"no-dict-update keeps dictionary hash stable with new English token")
    manifest=prep.prepare(a,report); payload=json.loads(manifest.read_text()); copied=payload["prepared_files"]
    fails+=ok(all(x["source_sha256"]==x["destination_sha256"] and x["source_size"]==x["destination_size"] and Path(x["destination_path"]).is_file() and not Path(x["destination_path"]).is_symlink() for x in copied) and any(x["kind"]=="run_local_dict" for x in copied),"prepare records regular source/destination hash evidence and run-local dict")
    try: prep.prepare(a,report); duplicate=False
    except FileExistsError: duplicate=True
    fails+=ok(duplicate,"existing target rejected")
    try: prep.finalize(a,report); incomplete=False
    except ValueError: incomplete=True
    fails+=ok(incomplete,"missing rerun set blocks ready")
    a_stems=args(source,legacy,dictionary,root/"run-stems"); r_stems=prep.inspect(source,legacy); prep.prepare(a_stems,r_stems)
    (a_stems.run_root/"rerun_stems.txt").write_text("wrong\n")
    try: prep.finalize(a_stems,r_stems); stems_tamper=False
    except ValueError: stems_tamper=True
    fails+=ok(stems_tamper,"rerun_stems content/hash tamper blocks ready")
    a_dict=args(source,legacy,dictionary,root/"run-dict"); r_dict=prep.inspect(source,legacy); prep.prepare(a_dict,r_dict)
    before=prep.sha256(a_dict.run_root/"dict"/"mfa_ipa.dict"); (a_dict.run_root/"dict"/"mfa_ipa.dict").write_text("newenglish newenglish\n")
    try: prep.finalize(a_dict,r_dict); dict_drift=False
    except ValueError: dict_drift=True
    fails+=ok(before != prep.sha256(a_dict.run_root/"dict"/"mfa_ipa.dict") and dict_drift,"run-local dictionary drift is detected; rendered CTC prevents it")
    a_source=args(source,legacy,dictionary,root/"run-source"); r_source=prep.inspect(source,legacy); prep.prepare(a_source,r_source)
    old_text=(source/"bad.txt").read_text(); (source/"bad.txt").write_text("source tamper\n")
    try: prep.finalize(a_source,r_source); source_tamper=False
    except ValueError: source_tamper=True
    (source/"bad.txt").write_text(old_text)
    fails+=ok(source_tamper,"source evidence tamper blocks ready")
    a_escape=args(source,legacy,dictionary,root/"run-escape"); r_escape=prep.inspect(source,legacy); prep.prepare(a_escape,r_escape)
    a_escape.rerun_ctc=root/"outside-rerun"; bundle(a_escape.rerun_ctc,"bad")
    try: prep.finalize(a_escape,r_escape); escaped=False
    except ValueError: escaped=True
    fails+=ok(escaped,"rerun path escape is rejected")
    a_link=args(source,legacy,dictionary,root/"run-link"); r_link=prep.inspect(source,legacy); prep.prepare(a_link,r_link)
    target=a_link.run_root/"ctc_rerun_output"; target.mkdir(); (target/"bad.lab").symlink_to(source/"bad.txt")
    try: prep.finalize(a_link,r_link); linked=False
    except ValueError: linked=True
    fails+=ok(linked,"rerun symlink is rejected")
    # Separate roots avoid test cleanup/mutation affecting a ready transaction.
    a2=args(source,legacy,dictionary,root/"run2"); r2=prep.inspect(source,legacy); prep.prepare(a2,r2); bundle(a2.run_root/"ctc_rerun_output","bad"); bundle(a2.run_root/"ctc_rerun_output","extra")
    try: prep.finalize(a2,r2); extra=False
    except ValueError: extra=True
    fails+=ok(extra,"rerun extra stem blocks before copy")
    a3=args(source,legacy,dictionary,root/"run3"); r3=prep.inspect(source,legacy); prep.prepare(a3,r3); bundle(a3.run_root/"ctc_rerun_output","bad")
    (a3.run_root/"audio_view"/"good1.wav").write_bytes(b"tamper")
    try: prep.finalize(a3,r3); tamper=False
    except ValueError: tamper=True
    fails+=ok(tamper,"prepared destination tamper blocks ready")
    a4=args(source,legacy,dictionary,root/"run4"); r4=prep.inspect(source,legacy); prep.prepare(a4,r4); bundle(a4.run_root/"ctc_rerun_output","bad")
    evidence=prep.finalize(a4,r4); prep.verify_ready(a4,r4)
    fails+=ok(json.loads(evidence.read_text())["state"]=="ready" and prep.bundle_stems(a4.run_root/"ctc_ready",a4.run_root)=={"good1","good2","bad"},"final exact CTC/audio sets verify ready")
    original=evidence.read_text()
    def evidence_rejected(mutator,label):
      nonlocal fails
      value=json.loads(original); mutator(value); evidence.write_text(json.dumps(value))
      try: prep.verify_ready(a4,r4); rejected=False
      except ValueError: rejected=True
      evidence.write_text(original); fails+=ok(rejected,label)
    evidence_rejected(lambda x:x["artifacts"].pop("good1"),"ready evidence rejects omitted artifact")
    evidence_rejected(lambda x:x["artifacts"].update({"extra":{}}),"ready evidence rejects extra artifact")
    evidence_rejected(lambda x:x["artifacts"]["good1"].__setitem__("audio",x["artifacts"]["good2"]["audio"]),"ready evidence rejects cross-stem path")
    evidence_rejected(lambda x:x["artifacts"]["good1"].__setitem__("reference",x["artifacts"]["good2"]["reference"]),"ready evidence rejects cross-stem reference")
    evidence_rejected(lambda x:x["rerun_files"].pop(),"ready evidence rejects omitted rerun record")
    evidence_rejected(lambda x:x["rerun_files"][0].__setitem__("destination_path",x["rerun_files"][0]["source_path"]),"ready evidence rejects misdirected rerun record")
    (a4.run_root/"ctc_ready"/"bad.lab").write_text("tamper\n")
    try: prep.verify_ready(a4,r4); ready_tamper=False
    except ValueError: ready_tamper=True
    fails+=ok(ready_tamper,"read-only ready verifier rejects artifact tamper")
    badargs=args(source,legacy,dictionary,root/"unused"); badargs.expected_txts=4
    try: prep.enforce_counts(report,badargs); count_fail=False
    except ValueError: count_fail=True
    fails+=ok(count_fail,"production count/exact missing gate is configurable and strict")
  return fails
def v4_main():
  """Focused tempdir-only v4 transaction and independent-verifier regressions."""
  fails=0
  with tempfile.TemporaryDirectory() as td:
    root=Path(td); source=root/'source'; source.mkdir(); legacy=root/'unused'; legacy.mkdir(); dictionary=root/'dict'; dictionary.write_text('ni3 n i3\n')
    for stem in ('a','b'):
      wav(source/f'{stem}.wav'); (source/f'{stem}.txt').write_text('你\n')
    report=prep.inspect(source); a=args(source,legacy,dictionary,root/'run'); a.expected_wavs=a.expected_txts=a.expected_authoritative=2; a.expected_missing_refs=a.expected_txt_only=0; a.expected_missing_stems=[]
    fails+=ok(report['schema']=='hecheng-english-ctc-ready-v4' and report['action_counts']=={'acoustic_rerun':2} and report['taxonomy']==[{'stem':'a','reason':'legacy_audio_provenance_unbound','action':'acoustic_rerun'},{'stem':'b','reason':'legacy_audio_provenance_unbound','action':'acoustic_rerun'}], 'v4 inspect is legacy-free all-fresh taxonomy')
    final_ctc=root/'final-ctc'; final_ctc.mkdir(); final_audio=root/'final-audio'; final_audio.mkdir()
    for stem,duration in (('one',1.0),('grid',9.44)):
      with wave.open(str(final_audio/f'{stem}.wav'),'wb') as h: h.setnchannels(1); h.setsampwidth(2); h.setframerate(16000); h.writeframes(b'\0\0'*round(duration*16000))
      ctc.write_textgrid([{'word':'ria','start':.1,'end':duration-.1}],duration,final_ctc/f'{stem}.TextGrid',pauses=[])
      (final_ctc/f'{stem}.lab').write_text('ria\n'); (final_ctc/f'{stem}_tokens.jsonl').write_text(json.dumps({'word':'ria','start_s':.1,'end_s':duration-.1,'start_ms':100,'end_ms':round((duration-.1)*1000,1)})+'\n'); (final_ctc/f'{stem}_punct.json').write_text('[]'); (final_ctc/f'{stem}_text_cn.txt').write_text('ria\n'); (final_ctc/f'{stem}_text_raw.txt').write_text('ria\n')
    ctc._rebuild_final_manifest(final_ctc,final_audio); final_manifest=json.loads((final_ctc/'manifest.json').read_text())
    fails+=ok([(x['audio'].split('/')[-1],x['duration_s'],x['n_words'],x['_words'][0]['word']) for x in final_manifest]==[('grid.wav',9.44,1,'ria'),('one.wav',1.0,1,'ria')], 'final manifest uses normalized ria tokens and physical 1.00/9.44 WAV axes')
    command=prep.render_rerun_command(a); source_code=(PROJECT_ROOT/'scripts'/'ctc_prealign.py').read_text()
    fails+=ok('--require-fresh-output' in command and '--overwrite' not in command and '"--overwrite",' not in source_code[source_code.index('child_argv += ['):source_code.index('# Copy dict to shard dir')], 'v4 parent/child CTC argv has fresh gate and no overwrite')
    overlap=root/'overlap.jsonl'; overlap.write_text('{"word":"a","start_s":0,"end_s":0.7,"start_ms":0,"end_ms":700}\n{"word":"b","start_s":0.2,"end_s":0.8,"start_ms":200,"end_ms":800}\n')
    punct=root/'punct.json'; punct.write_text('[{"word":"，","start_s":0,"end_s":0.9,"start_ms":0,"end_ms":900},{"word":"。","start_s":0.2,"end_s":0.4,"start_ms":200,"end_ms":400}]')
    empty=root/'empty.json'; empty.write_text('[]')
    fails+=ok(len(prep._v4_json(overlap,False,1.0))==2 and len(prep._v4_json(punct,True,1.0))==2 and prep._v4_json(empty,True,1.0)==[], 'token/punct overlap and no-punct [] are accepted')
    prep.prepare(a,report); rerun=a.run_root/'ctc_rerun_output'; bundle(rerun,'a'); bundle(rerun,'b')
    for stem in ('a','b'): (rerun/f'{stem}_ref.txt').write_text((source/f'{stem}.txt').read_text())
    manifest=[{'audio':str((a.run_root/'audio_view'/f'{stem}.wav').resolve()),'textgrid':str((rerun/f'{stem}.TextGrid').resolve()),'lab':str((rerun/f'{stem}.lab').resolve()),'duration_s':1.0,'n_words':1,'_words':[{'word':'ni3','start':0.0}]} for stem in ('a','b')]
    (rerun/'manifest.json').write_text(json.dumps(manifest)); (rerun/'summary.txt').write_text('Files: 2 total, 2 OK, 0 failed\n'); (rerun/'.ctc_normalized').write_text('reference-authority-v3-safe-transcript\n')
    model_digest, model_files = compute_model_tree_digest(Path(a.asr_model))
    write_ctc_run_receipt(rerun, [a.asr_python], a.asr_python, Path(a.asr_model),
                          model_digest, model_files, a.run_root/'dict'/'mfa_ipa.dict',
                          prep.sha256(a.run_root/'dict'/'mfa_ipa.dict'), ['a', 'b'], ['a', 'b'])
    evidence=prep.finalize(a,report); prep.verify_ready(a,report)
    verifier_command=[sys.executable,str(PROJECT_ROOT/'scripts'/'verify_hecheng_english_ctc_ready_v4.py'),'--run-root',str(a.run_root),'--source-dir',str(source),'--dictionary-source',str(dictionary),'--asr-python',a.asr_python,'--asr-model',a.asr_model]
    independent=subprocess.run(verifier_command,text=True,capture_output=True)
    payload=json.loads(evidence.read_text())
    fails+=ok(independent.returncode==0 and payload['stem_count']==2 and payload['artifacts']['a']['origin_action']=='acoustic_rerun' and payload['artifacts']['a']['audio']['sha256']==payload['artifacts']['a']['authoritative_audio']['sha256'], 'v4 finalize and independent verifier bind authoritative audio')
    original=evidence.read_text(); payload['authoritative_stems']=['b','a']; evidence.write_text(json.dumps(payload))
    bad=subprocess.run(verifier_command,text=True,capture_output=True)
    evidence.write_text(original); fails+=ok(bad.returncode!=0,'independent verifier rejects reordered evidence stems')
    token=a.run_root/'ctc_ready'/'a_tokens.jsonl'; before=token.read_text(); token.write_text(before.replace('"start_s": 0','"start_s": NaN'))
    bad=subprocess.run(verifier_command,text=True,capture_output=True); token.write_text(before)
    fails+=ok(bad.returncode!=0,'independent verifier rejects nonfinite token timing')
    manifest_path=a.run_root/'prepare_manifest.json'; manifest_before=manifest_path.read_text(); manifest_value=json.loads(manifest_before); manifest_value['prepared_files_sha256']='bad'; manifest_path.write_text(json.dumps(manifest_value))
    bad=subprocess.run(verifier_command,text=True,capture_output=True); manifest_path.write_text(manifest_before)
    fails+=ok(bad.returncode!=0 and 'prepared files' in bad.stderr,'independent verifier rejects prepared manifest hash tamper')
    (a.run_root/'audio_view'/'extra.wav').write_bytes(b'extra')
    bad=subprocess.run(verifier_command,text=True,capture_output=True); (a.run_root/'audio_view'/'extra.wav').unlink()
    fails+=ok(bad.returncode!=0 and 'namespace extra/missing' in bad.stderr,'independent verifier rejects audio namespace extra')
    (rerun/'extra.bin').write_bytes(b'extra'); bad=subprocess.run(verifier_command,text=True,capture_output=True); (rerun/'extra.bin').unlink()
    fails+=ok(bad.returncode!=0 and 'namespace extra/missing' in bad.stderr,'independent verifier rejects rerun namespace extra')
    payload=json.loads(original); payload['artifacts']['a']['authoritative_audio']=payload['artifacts']['b']['authoritative_audio']; evidence.write_text(json.dumps(payload))
    bad=subprocess.run(verifier_command,text=True,capture_output=True); evidence.write_text(original)
    fails+=ok(bad.returncode!=0 and 'authority substitution' in bad.stderr,'independent verifier rejects authoritative cross-stem substitution')
    marker=rerun/'.ctc_normalized'; marker_before=marker.read_text(); marker.write_text('wrong\n'); bad=subprocess.run(verifier_command,text=True,capture_output=True); marker.write_text(marker_before)
    fails+=ok(bad.returncode!=0 and 'normalization marker' in bad.stderr,'independent verifier rejects normalization marker drift')
    summary=rerun/'summary.txt'; summary_before=summary.read_text(); summary.write_text('Files: 2 total, 1 OK, 1 failed\n'); bad=subprocess.run(verifier_command,text=True,capture_output=True); summary.write_text(summary_before)
    fails+=ok(bad.returncode!=0 and 'rerun summary' in bad.stderr,'independent verifier rejects rerun summary failure count')
    run_dict=a.run_root/'dict'/'mfa_ipa.dict'; dict_before=run_dict.read_text(); run_dict.write_text(dict_before+'drift x\n'); bad=subprocess.run(verifier_command,text=True,capture_output=True); run_dict.write_text(dict_before)
    fails+=ok(bad.returncode!=0 and ('prepared copy evidence' in bad.stderr or 'dictionary evidence' in bad.stderr),'independent verifier rejects run-local dictionary drift')
    manifest_value=json.loads(manifest_before); manifest_value['rerun_command'][0]='wrong-python'; manifest_path.write_text(json.dumps(manifest_value)); bad=subprocess.run(verifier_command,text=True,capture_output=True); manifest_path.write_text(manifest_before)
    fails+=ok(bad.returncode!=0 and 'rerun command' in bad.stderr,'independent verifier rejects ASR command identity drift')
    alias_root=root/'run-alias'; alias_args=args(source,legacy,dictionary,alias_root); alias_args.expected_wavs=alias_args.expected_txts=alias_args.expected_authoritative=2; alias_args.expected_missing_refs=alias_args.expected_txt_only=0; alias_args.expected_missing_stems=[]; prep.prepare(alias_args,report)
    alias_audio=alias_root/'audio_view'/'a.wav'; alias_audio.unlink(); os.link(source/'a.wav',alias_audio)
    try: prep._v4_load(alias_args,report); alias_rejected=False
    except ValueError: alias_rejected=True
    fails+=ok(alias_rejected,'preparer rejects prepared-copy hardlink substitution')
    r2=root/'run-malformed'; bargs=args(source,legacy,dictionary,r2); bargs.expected_wavs=bargs.expected_txts=bargs.expected_authoritative=2; bargs.expected_missing_refs=bargs.expected_txt_only=0; bargs.expected_missing_stems=[]; prep.prepare(bargs,report); out=r2/'ctc_rerun_output'; bundle(out,'a'); bundle(out,'b'); malformed_textgrid(out/'a.TextGrid')
    for stem in ('a','b'): (out/f'{stem}_ref.txt').write_text((source/f'{stem}.txt').read_text())
    manifest=[{'audio':str((r2/'audio_view'/f'{stem}.wav').resolve()),'textgrid':str((out/f'{stem}.TextGrid').resolve()),'lab':str((out/f'{stem}.lab').resolve()),'duration_s':1.0,'n_words':1,'_words':[{'word':'ni3','start':0.0}]} for stem in ('a','b')]
    (out/'manifest.json').write_text(json.dumps(manifest)); (out/'summary.txt').write_text('Files: 2 total, 2 OK, 0 failed\n'); (out/'.ctc_normalized').write_text('reference-authority-v3-safe-transcript\n')
    try: prep.finalize(bargs,report); rejected=False
    except ValueError: rejected=True
    fails+=ok(rejected,'v4 finalize rejects malformed rerun TextGrid')
  return fails
if __name__=="__main__": raise SystemExit(v4_main())
