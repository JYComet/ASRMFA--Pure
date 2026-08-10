#!/usr/bin/env python3
"""Independent fail-closed verifier for v4 CTC-ready evidence.

It intentionally does not import the preparation script or any of its parser
or transform helpers.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, sys, wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

SCHEMA="hecheng-english-ctc-ready-v4"; SIGNATURE="ctc-ready-independent-v1"; SUFFIXES=(".TextGrid",".lab","_tokens.jsonl","_punct.json","_text_cn.txt","_text_raw.txt"); REF_SUFFIX="_ref.txt"; REQUIRED_SUFFIXES=SUFFIXES+(REF_SUFFIX,); TOL=.003; DOMAIN_TOL=.000001; NVV_MODE="reference_only"; ASR_NVV_BIAS=False; CONTENT_AUTHORITY="reference"
SOURCE=Path('/mnt/Raw/新版合成英文数据')
DEFAULT_DICT=Path(__file__).resolve().parent.parent/'dict'/'mfa_ipa.dict'
DEFAULT_ASR_PYTHON='/home/user/miniconda3/envs/asr/bin/python'
DEFAULT_ASR_MODEL='/mnt/local_E/nvvasr_standalone/models/Multilingual-NVASR'
NVV_TO_MFA={'Breathing':'BREATHING','Laughter':'LAUGHTER','Burp':'BURP','Cough':'COUGH','Crying':'CRYING','Groan':'GROAN','Hiss':'HISS','Hum':'HUM','Shh':'SHH','Sigh':'SIGH','Sneeze':'SNEEZE','Sniff':'SNIFF','Snore':'SNORE','Tsk':'TSK','Uhm':'UHM','Whistle':'WHISTLE','Yawn':'YAWN','Question-yi':'QUESTION-YI','Question-en':'QUESTION-EN','Question-oh':'QUESTION-OH','Question-ah':'QUESTION-AH','Question-ei':'QUESTION-EI','Question-huh':'QUESTION-HUH','Surprise-oh':'SURPRISE-OH','Surprise-ah':'SURPRISE-AH','Surprise-wa':'SURPRISE-WA','Surprise-yo':'SURPRISE-YO','Confirmation-en':'CONFIRMATION-EN','Dissatisfaction-hnn':'DISSATISFACTION-HNN'}
NVV_NAMES=set(NVV_TO_MFA.values())
PUNCT_MAP={',':'，','.':'。','?':'？','!':'！',';':'；',':':'：'}
PUNCT=set('，。！？、；：…')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def stable(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def _reference_projection(text):
 try:
  from pypinyin import lazy_pinyin,Style
 except Exception as e:
  raise ValueError('pypinyin is required for independent reference projection') from e
 pattern=re.compile(r'\[[A-Za-z][^\]]*\]|<[A-Za-z][^>]*>|[A-Za-z]+(?:[-\'][A-Za-z]+)?|\d+|[一-鿿㐀-䶿]|[^\s]')
 lexical=[]; punctuation=[]; all_items=[]
 for match in pattern.finditer(text.strip()):
  raw=match.group(0); upper=raw.strip('<>').upper()
  if raw.startswith('[') and raw.endswith(']'):
   label=NVV_TO_MFA.get(raw[1:-1],raw[1:-1].upper().replace(' ','-'))
   if label not in NVV_NAMES: raise ValueError(f'unknown reference NVV: {raw}')
   lexical.append(label); all_items.append(('lexical',label)); continue
  if raw.startswith('<') and raw.endswith('>'):
   if upper in NVV_NAMES:
    lexical.append(upper); all_items.append(('lexical',upper)); continue
   if re.fullmatch(r'<sp[0-9]+>',raw,re.I):
    all_items.append(('structural',raw.lower())); continue
   raise ValueError(f'unknown reference tag: {raw}')
  if len(raw)==1 and raw in PUNCT_MAP: raw=PUNCT_MAP[raw]
  if raw in PUNCT:
   punctuation.append(raw); all_items.append(('punct',raw)); continue
  if '一'<=raw<='鿿' or '㐀'<=raw<='䶿':
   py=lazy_pinyin(raw,style=Style.TONE3,neutral_tone_with_five=True,errors='default')
   if not py or not re.fullmatch(r'[a-z]+[1-5]',py[0]): raise ValueError(f'CJK has no tone pinyin: {raw}')
   lexical.append(py[0]); all_items.append(('lexical',py[0])); continue
  if raw.isalpha() or raw.isdigit():
   lexical.append(raw.lower()); all_items.append(('lexical',raw.lower())); continue
  raise ValueError(f'unsupported reference content: {raw}')
 return lexical,punctuation,all_items
def _actual_projection(root,stem,reference):
 expected_lex,expected_punct,items=_reference_projection(reference)
 lab=(root/(stem+'.lab')).read_text(encoding='utf-8-sig').strip().split()
 structural=[x.strip().lower() for x in lab if x.strip().lower().startswith('<sp')]
 if any(x != '<sp1>' for x in structural) or structural.count('<sp1>') > 1 or (structural and lab[0].strip().lower() != '<sp1>'):
  raise ValueError('invalid sentence-initial sp1 grammar')
 actual_lex=[x.strip() for x in lab if x.strip() and not x.strip().lower().startswith('<sp')]
 if any(x.strip().lower() in {'unknown','spn','<unk>','<spn>'} for x in actual_lex):
  raise ValueError('unknown/spn is not a reference lexical token')
 actual_norm=[x.upper() if x.upper() in NVV_NAMES else x.lower() for x in actual_lex]
 if actual_norm!=expected_lex: raise ValueError(f'reference lexical projection mismatch: expected {expected_lex!r}, got {actual_norm!r}')
 expected_pinyin=sum(1 for x in expected_lex if re.fullmatch(r'[a-z]+[1-5]',x))
 actual_pinyin=sum(1 for x in actual_norm if re.fullmatch(r'[a-z]+[1-5]',x))
 if expected_pinyin != actual_pinyin:
  raise ValueError('0 pinyin or CJK lexical projection mismatch')
 try: punct_data=json.loads((root/(stem+'_punct.json')).read_text(encoding='utf-8-sig'))
 except Exception as e: raise ValueError('bad reference punctuation sidecar') from e
 actual_punct=[str(x.get('word','')) for x in punct_data]
 if actual_punct!=expected_punct: raise ValueError(f'reference punctuation mismatch: expected {expected_punct!r}, got {actual_punct!r}')
 canonical=reference.strip()+'\n'
 for suffix in ('_ref.txt','_text_raw.txt','_text_cn.txt'):
  if (root/(stem+suffix)).read_text(encoding='utf-8-sig')!=canonical: raise ValueError(f'reference sidecar content mismatch: {suffix}')
def regular(p):
 if p.is_symlink() or not p.is_file():raise ValueError(f'not regular: {p}')
def ev(p,wav=False):
 regular(p); r={'path':str(p.resolve()),'size':p.stat().st_size,'sha256':digest(p)}
 if wav:
  with wave.open(str(p),'rb') as h:fr,ra,ch=h.getnframes(),h.getframerate(),h.getnchannels()
  if min(fr,ra,ch)<=0:raise ValueError('bad WAV')
  r['wav']={'frames':fr,'sample_rate':ra,'channels':ch,'duration_s':fr/ra}
 return r
def timing(p,punct,dur):
 try:x=json.loads(p.read_text(encoding='utf-8-sig')) if punct else [json.loads(z) for z in p.read_text(encoding='utf-8-sig').splitlines() if z.strip()]
 except Exception as e:raise ValueError('bad timing JSON') from e
 if not isinstance(x,list):raise ValueError('timing not list')
 last_start=last_end=-math.inf
 for i,z in enumerate(x):
  try:a,b,am,bm=map(float,(z['start_s'],z['end_s'],z['start_ms'],z['end_ms']))
  except Exception as e:raise ValueError(f'timing {i}') from e
  if not isinstance(z,dict) or (not punct and not str(z.get('word','')).strip()) or not all(math.isfinite(q) for q in(a,b,am,bm)) or b<=a or a<-DOMAIN_TOL or b>dur+DOMAIN_TOL or abs(a*1000-am)>.51 or abs(b*1000-bm)>.51 or (not punct and (a+DOMAIN_TOL<last_start or b+DOMAIN_TOL<last_end)):raise ValueError(f'timing invalid {i}')
  if not punct:last_start,last_end=a,b
 return x
def uq(s):
 s=s.split('=',1)[1].strip()
 if not(s.startswith('"') and s.endswith('"')):raise ValueError('bad quote')
 return s[1:-1].replace('""','"')
def num(s,key):
 if not s.startswith(key+' = '):raise ValueError('expected '+key)
 x=float(s.split('=',1)[1]);
 if not math.isfinite(x):raise ValueError('nonfinite')
 return x
def tgparse(p):
 ls=[x.strip() for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]; i=0
 def take(x):
  nonlocal i
  if i>=len(ls) or ls[i]!=x:raise ValueError('TextGrid grammar '+x)
  i+=1
 def pref(x):
  nonlocal i
  if i>=len(ls) or not ls[i].startswith(x):raise ValueError('TextGrid grammar '+x)
  z=ls[i];i+=1;return z
 take('File type = "ooTextFile"');take('Object class = "TextGrid"');xmin=num(pref('xmin = '),'xmin');xmax=num(pref('xmax = '),'xmax');take('tiers? <exists>');take('size = 2');take('item []:');tiers=[]
 for n in (1,2):
  take(f'item [{n}]:');take('class = "IntervalTier"'); name=uq(pref('name = '));a=num(pref('xmin = '),'xmin');b=num(pref('xmax = '),'xmax'); size=int(pref('intervals: size = ').split('=',1)[1]);
  if size < 0:raise ValueError('negative interval size')
  ivs=[]
  for j in range(1,size+1):
   take(f'intervals [{j}]:');ivs.append((num(pref('xmin = '),'xmin'),num(pref('xmax = '),'xmax'),uq(pref('text = '))))
  tiers.append((name,a,b,ivs))
 if i!=len(ls):raise ValueError('TextGrid tail')
 return xmin,xmax,tiers
def validate_bundle(root,stem,audio):
 for s in SUFFIXES:regular(root/(stem+s))
 m=ev(audio,True)['wav'];tok=timing(root/(stem+'_tokens.jsonl'),False,m['duration_s']);timing(root/(stem+'_punct.json'),True,m['duration_s']);a,b,tiers=tgparse(root/(stem+'.TextGrid'))
 if (root/(stem+'.lab')).read_text(encoding='utf-8-sig').strip().split() != [str(x['word']) for x in tok]:raise ValueError('lab/token transcript')
 if [x[0] for x in tiers]!=['words','pauses'] or abs(a)>DOMAIN_TOL or abs(b-m['duration_s'])>DOMAIN_TOL:raise ValueError('TextGrid top/domain')
 for _,x,y,ivs in tiers:
  if abs(x-a)>DOMAIN_TOL or abs(y-b)>DOMAIN_TOL:raise ValueError('tier domain')
  last=a
  for q,r,_ in ivs:
   if r<=q or q<-DOMAIN_TOL or r>m['duration_s']+DOMAIN_TOL or q+DOMAIN_TOL<last:raise ValueError('TextGrid interval')
   last=r
 words=[x for x in tiers[0][3] if x[2].strip()]
 if len(words)!=len(tok) or any(x[2].strip()!=str(y['word']).strip() or abs(x[0]-float(y['start_s']))>TOL for x,y in zip(words,tok)):raise ValueError('word/token')
def rerun_manifest(root,stems):
 if (root/'ctc_rerun_output'/'.ctc_normalized').read_text(encoding='utf-8')!='reference-authority-v3-safe-transcript\n':raise ValueError('normalization marker')
 summary=(root/'ctc_rerun_output'/'summary.txt').read_text(encoding='utf-8'); match=re.search(r'^Files:\s+(\d+)\s+total,\s+(\d+)\s+OK,\s+(\d+)\s+failed$',summary,re.M)
 if not match or tuple(map(int,match.groups()))!=(len(stems),len(stems),0):raise ValueError('rerun summary')
 try:entries=json.loads((root/'ctc_rerun_output'/'manifest.json').read_text(encoding='utf-8'))
 except Exception as e:raise ValueError('rerun manifest') from e
 if not isinstance(entries,list):raise ValueError('rerun manifest list')
 seen=[]
 for x in entries:
  if not isinstance(x,dict):raise ValueError('rerun manifest entry')
  audio=Path(x.get('audio',''));s=audio.stem;seen.append(s); rerun=root/'ctc_rerun_output'
  if str(audio.resolve())!=str((root/'audio_view'/(s+'.wav')).resolve()) or x.get('textgrid')!=str((rerun/(s+'.TextGrid')).resolve()) or x.get('lab')!=str((rerun/(s+'.lab')).resolve()):raise ValueError('rerun manifest paths')
  dur=ev(audio,True)['wav']['duration_s'];tok=timing(rerun/(s+'_tokens.jsonl'),False,dur);a,b,tiers=tgparse(rerun/(s+'.TextGrid'));words=[q for q in tiers[0][3] if q[2].strip()];wm=x.get('_words')
  if not isinstance(wm,list) or x.get('n_words')!=len(tok) or len(wm)!=len(tok) or not math.isfinite(float(x.get('duration_s',math.nan))) or abs(float(x['duration_s'])-dur)>DOMAIN_TOL:raise ValueError('rerun manifest timing/count')
  for w,t,iv in zip(wm,tok,words):
   if not isinstance(w,dict) or str(w.get('word','')).strip()!=str(t['word']).strip() or abs(float(w.get('start',math.nan))-float(t['start_s']))>TOL or abs(iv[0]-float(t['start_s']))>TOL:raise ValueError('rerun manifest words')
 if seen!=stems or len(seen)!=len(set(seen)):raise ValueError('rerun manifest coverage')
def source_inventory(source):
 if not source.is_dir() or source.is_symlink():raise ValueError('source directory')
 def index(suffix):
  out={};dup=set()
  for base,dirs,names in os.walk(source):
   if any((Path(base)/d).is_symlink() for d in dirs):raise ValueError('source symlink directory')
   for name in names:
    if name.lower().endswith(suffix):
     p=Path(base)/name;st=name[:-len(suffix)]
     if p.is_symlink():raise ValueError('source symlink file')
     if st in out:dup.add(st);out.pop(st,None)
     elif st not in dup:out[st]=p
  if dup:raise ValueError('source duplicate stems')
  return out
 wavs=index('.wav');txts=index('.txt');ws,ts=set(wavs),set(txts);stems=sorted(ws&ts);missing=sorted(ws-ts);only=sorted(ts-ws)
 if source.resolve()==SOURCE.resolve() and (len(wavs)!=54000 or len(txts)!=53998 or len(stems)!=53998 or missing!=['024198_杂谈互动_数据里程牌庆祝','036000_弹幕互动_回应吐槽弹幕'] or only):raise ValueError('authoritative inventory counts')
 taxonomy=[{'stem':s,'reason':'legacy_audio_provenance_unbound','action':'acoustic_rerun'} for s in stems]
 report={'schema':SCHEMA,'source_dir':str(source.resolve()),'wav_count':len(wavs),'txt_count':len(txts),'stem_count':len(stems),'authoritative_stems':stems,'missing_reference':missing,'txt_only':only,'wav_paths':{s:str(wavs[s].resolve()) for s in stems},'txt_paths':{s:str(txts[s].resolve()) for s in stems},'final_audio_axis':'authoritative_wav','padding_policy':'forbidden','action_counts':{'acoustic_rerun':len(stems)},'taxonomy':taxonomy,'taxonomy_sha256':stable(taxonomy)}
 report['inventory_sha256']=stable(report)
 return wavs,txts,stems,missing,only,report
def verify(run_root,source_dir=SOURCE,dictionary_source=DEFAULT_DICT,
           asr_python=DEFAULT_ASR_PYTHON,asr_model=DEFAULT_ASR_MODEL):
 root=run_root.resolve(); e=json.loads((root/'ctc_ready_evidence.json').read_text(encoding='utf-8')); stems=e.get('authoritative_stems')
 expected_keys={'schema','state','independent_verifier_signature','prepare_manifest_sha256','inventory_sha256','stem_count','authoritative_stems','missing_reference','txt_only','final_audio_axis','padding_policy','action_counts','taxonomy','taxonomy_sha256','nvv_mode','asr_nvv_bias','content_authority','roots','source_dictionary','run_local_dictionary','artifacts','rerun_files','rerun_files_sha256','asr_model_path','asr_model_tree_digest','asr_model_files','ctc_run_receipt_digest','pipeline_accounting_receipt'}
 if set(e)!=expected_keys:raise ValueError('evidence top-level keys')
 if e.get('schema')!=SCHEMA or e.get('state')!='ready' or e.get('independent_verifier_signature')!=SIGNATURE or e.get('final_audio_axis')!='authoritative_wav' or e.get('padding_policy')!='forbidden' or e.get('nvv_mode')!=NVV_MODE or e.get('asr_nvv_bias') is not ASR_NVV_BIAS or e.get('content_authority')!=CONTENT_AUTHORITY:raise ValueError('schema/axis/reference-only metadata')
 wavs,txts,expected_stems,expected_missing,expected_only,source_report=source_inventory(Path(source_dir))
 if not isinstance(stems,list) or stems!=expected_stems or e.get('inventory_sha256')!=source_report['inventory_sha256'] or e.get('stem_count')!=len(expected_stems) or e.get('missing_reference')!=expected_missing or e.get('txt_only')!=expected_only or e.get('action_counts')!={'acoustic_rerun':len(expected_stems)}:raise ValueError('stem/action list')
 # Frozen source-denominator receipt: missing-reference WAVs are explicit
 # exclusions and can never enter the strict ready set.
 try:
  from pipeline_utils import (PIPELINE_ACCOUNTING_SCHEMA,
                              read_pipeline_accounting_receipt,
                              validate_pipeline_accounting_receipt)
  binding=e.get('pipeline_accounting_receipt')
  if not isinstance(binding,dict) or binding.get('schema')!=PIPELINE_ACCOUNTING_SCHEMA: raise ValueError('pipeline accounting receipt binding')
  receipt_path=Path(binding.get('path',''))
  if receipt_path.resolve() != (root/'.pipeline_run_receipt_v2.json').resolve(): raise ValueError('pipeline accounting receipt path')
  receipt=read_pipeline_accounting_receipt(receipt_path)
  if digest(receipt_path)!=binding.get('sha256'): raise ValueError('pipeline accounting receipt hash')
  receipt_errors=validate_pipeline_accounting_receipt(receipt)
  if receipt_errors: raise ValueError('pipeline accounting receipt: '+ '; '.join(receipt_errors))
  excluded={x['stem'] for x in receipt['exclusions']}; source=set(receipt['source']['stems']); eligible=set(receipt['eligible']['stems'])
  if source != set(wavs) or eligible != set(expected_stems) or excluded != set(expected_missing) or source != eligible | excluded: raise ValueError('pipeline accounting source conservation')
 except (OSError,TypeError,ValueError,KeyError,json.JSONDecodeError) as ex:
  raise ValueError(str(ex)) from ex
 if not isinstance(e.get('taxonomy'),list) or e.get('taxonomy_sha256')!=hashlib.sha256(json.dumps(e['taxonomy'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest() or e['taxonomy'] != [{'stem':s,'reason':'legacy_audio_provenance_unbound','action':'acoustic_rerun'} for s in stems]:raise ValueError('taxonomy')
 roots=e.get('roots',{}); expected={'run':str(root),'audio_view':str((root/'audio_view').resolve()),'reference_view':str((root/'reference_view').resolve()),'ctc_ready':str((root/'ctc_ready').resolve())}
 if roots!=expected:raise ValueError('roots')
 def exact_dir(folder,names):
  if folder.is_symlink() or not folder.is_dir():raise ValueError('namespace directory')
  actual=[]
  for p in folder.iterdir():
   if p.is_symlink() or not p.is_file():raise ValueError('namespace nonordinary')
   actual.append(p.name)
  if sorted(actual)!=sorted(names):raise ValueError('namespace extra/missing')
 exact_dir(root/'audio_view',[s+'.wav' for s in stems]);exact_dir(root/'reference_view',[s+'.txt' for s in stems]);exact_dir(root/'dict',['mfa_ipa.dict']);exact_dir(root/'ctc_ready',[s+q for s in stems for q in REQUIRED_SUFFIXES]);exact_dir(root/'ctc_rerun_output',[s+q for s in stems for q in REQUIRED_SUFFIXES]+['manifest.json','summary.txt','.ctc_normalized','.ctc_run_receipt.json'])
 try:m=json.loads((root/'prepare_manifest.json').read_text(encoding='utf-8'))
 except Exception as ex:raise ValueError('prepare manifest') from ex
 manifest_keys={'schema','state','inventory_sha256','stem_count','authoritative_stems','missing_reference','txt_only','final_audio_axis','padding_policy','action_counts','taxonomy','taxonomy_sha256','prepared_files','prepared_files_sha256','source_dictionary','run_local_dictionary','rerun_command','nvv_mode','asr_nvv_bias','content_authority','asr_model_path','asr_model_tree_digest','asr_model_files'}
 if set(m)!=manifest_keys | {'pipeline_accounting_receipt'} or m.get('schema')!=SCHEMA or m.get('state')!='awaiting_acoustic_rerun' or m.get('inventory_sha256')!=source_report['inventory_sha256'] or m.get('stem_count')!=len(stems) or m.get('authoritative_stems')!=stems or m.get('missing_reference')!=expected_missing or m.get('txt_only')!=expected_only or m.get('final_audio_axis')!='authoritative_wav' or m.get('padding_policy')!='forbidden' or m.get('action_counts')!=e['action_counts'] or m.get('taxonomy')!=e['taxonomy'] or m.get('taxonomy_sha256')!=e['taxonomy_sha256'] or m.get('source_dictionary')!=e['source_dictionary'] or m.get('run_local_dictionary')!=e['run_local_dictionary'] or m.get('nvv_mode')!=NVV_MODE or m.get('asr_nvv_bias') is not ASR_NVV_BIAS or m.get('content_authority')!=CONTENT_AUTHORITY or m.get('pipeline_accounting_receipt')!=e.get('pipeline_accounting_receipt'):raise ValueError('prepare manifest binding')
 expected_prepared=[]
 for s in stems:
  expected_prepared += [('audio_view',s,str(wavs[s].resolve()),str((root/'audio_view'/(s+'.wav')).resolve()),True),('reference_view',s,str(txts[s].resolve()),str((root/'reference_view'/(s+'.txt')).resolve()),False)]
 expected_prepared += [('run_local_dict','',str(Path(dictionary_source).resolve()),str((root/'dict'/'mfa_ipa.dict').resolve()),False)]
 actual_prepared=[]
 for x in m.get('prepared_files',[]):
  if set(x)!={'kind','stem','source','destination'}:raise ValueError('prepared record shape')
  actual_prepared.append((x.get('kind'),x.get('stem'),x.get('source',{}).get('path'),x.get('destination',{}).get('path'),x.get('kind')=='audio_view'))
  if x.get('source')!=ev(Path(x['source']['path']),x.get('kind')=='audio_view') or x.get('destination')!=ev(Path(x['destination']['path']),x.get('kind')=='audio_view') or os.path.samestat(Path(x['source']['path']).stat(),Path(x['destination']['path']).stat()):raise ValueError('prepared copy evidence')
 if actual_prepared!=expected_prepared or m.get('prepared_files_sha256')!=stable(m.get('prepared_files')):raise ValueError('prepared files')
 command=m.get('rerun_command'); expected_command=[str(asr_python),'scripts/ctc_prealign.py','--data-dir',str(root/'reference_view'),'--audio-dir',str(root/'audio_view'),'--pinyin-dir',str(root/'reference_view'),'--output-dir',str(root/'ctc_rerun_output'),'--model-path',str(asr_model),'--dict-path',str(root/'dict'/'mfa_ipa.dict'),'--all-gpus','--no-dict-update','--require-fresh-output','--no-nvv']
 if command!=expected_command or command.count('--no-nvv')!=1 or '--overwrite' in command:raise ValueError('rerun command')
 rerun_manifest(root,stems)
 # ── Model tree + receipt cross-check (Case 99 / R5) ──────────────
 try:
  from pipeline_utils import compute_model_tree_digest
 except ImportError:
  raise ValueError('cannot import model tree digest for provenance check')
 _model_p = Path(asr_model).resolve()
 _current_tree_digest, _current_tree_manifest = compute_model_tree_digest(_model_p)
 if m.get('asr_model_path') != str(_model_p) or m.get('asr_model_tree_digest') != _current_tree_digest or m.get('asr_model_files') != _current_tree_manifest:
  raise ValueError('ASR model tree does not match prepare freeze')
 # Receipt cross-check
 _receipt_path = root/'ctc_rerun_output'/'.ctc_run_receipt.json'
 if not _receipt_path.is_file():
  raise ValueError('missing CTC run receipt in ctc_rerun_output')
 try:
  _receipt = json.loads(_receipt_path.read_text(encoding='utf-8'))
 except Exception as ex:
  raise ValueError('invalid CTC run receipt') from ex
 if _receipt.get('schema') != 'ctc-run-receipt-v1':
  raise ValueError('CTC run receipt schema mismatch')
 _receipt_model_digest = _receipt.get('model', {}).get('tree_digest', '')
 if _receipt_model_digest != _current_tree_digest:
  raise ValueError('CTC run receipt model tree digest does not match current model tree')
 _receipt_input = sorted(_receipt.get('input_stems', []))
 _receipt_output = sorted(_receipt.get('output_stems', []))
 if _receipt_input != stems:
  raise ValueError('CTC run receipt input stems do not match expected')
 if _receipt_output != stems:
  raise ValueError('CTC run receipt output stems do not match expected')
 if (e.get('asr_model_path') != str(_model_p)
     or e.get('asr_model_tree_digest') != _current_tree_digest
     or e.get('asr_model_files') != _current_tree_manifest
     or e.get('ctc_run_receipt_digest') != digest(_receipt_path)):
  raise ValueError('ready evidence model/receipt binding mismatch')
 # ──────────────────────────────────────────────────────────────────
 art=e.get('artifacts');
 if not isinstance(art,dict) or sorted(art)!=stems:raise ValueError('artifacts')
 for s in stems:
  x=art[s]
  if set(x)!={'origin_action','audio','reference','authoritative_audio','authoritative_reference','ctc'} or x['origin_action']!='acoustic_rerun' or set(x['ctc'])!=set(REQUIRED_SUFFIXES):raise ValueError('artifact shape')
  pairs=[(x['audio'],root/'audio_view'/(s+'.wav'),True),(x['reference'],root/'reference_view'/(s+'.txt'),False),(x['authoritative_audio'],None,True),(x['authoritative_reference'],None,False)] + [(x['ctc'][q],root/'ctc_ready'/(s+q),False) for q in REQUIRED_SUFFIXES]
  for record,path,w in pairs:
   p=Path(record.get('path','')) if path is None else path
   if record!=ev(p,w) or (path is not None and record['path']!=str(path.resolve())):raise ValueError('artifact hash/path')
  if x['authoritative_audio']!=ev(wavs[s],True) or x['authoritative_reference']!=ev(txts[s]) or x['audio']['wav']!=x['authoritative_audio']['wav'] or x['audio']['sha256']!=x['authoritative_audio']['sha256'] or x['reference']['sha256']!=x['authoritative_reference']['sha256']:raise ValueError('authority substitution')
  if os.path.samestat((root/'audio_view'/(s+'.wav')).stat(),wavs[s].stat()) or os.path.samestat((root/'reference_view'/(s+'.txt')).stat(),txts[s].stat()):raise ValueError('copy inode alias')
  validate_bundle(root/'ctc_ready',s,root/'audio_view'/(s+'.wav'))
  reference_text=Path(txts[s]).read_text(encoding='utf-8-sig')
  _actual_projection(root/'ctc_ready',s,reference_text)
  _actual_projection(root/'ctc_rerun_output',s,reference_text)
  rerun_ref=root/'ctc_rerun_output'/(s+'_ref.txt'); reference=root/'reference_view'/(s+'.txt')
  if rerun_ref.read_bytes()!=reference.read_bytes() or os.path.samestat(rerun_ref.stat(),reference.stat()):raise ValueError('rerun reference')
 rerun=e.get('rerun_files'); expected=[]
 for s in stems:
  for q in REQUIRED_SUFFIXES:expected.append(('rerun_ctc',s,str((root/'ctc_rerun_output'/(s+q)).resolve()),str((root/'ctc_ready'/(s+q)).resolve())))
 actual=[(x.get('kind'),x.get('stem'),x.get('source',{}).get('path'),x.get('destination',{}).get('path')) for x in rerun] if isinstance(rerun,list) else []
 if actual!=expected or e.get('rerun_files_sha256')!=hashlib.sha256(json.dumps(rerun,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest():raise ValueError('rerun copy list')
 for x in rerun:
  if x.get('source')!=ev(Path(x['source']['path'])) or x.get('destination')!=ev(Path(x['destination']['path'])):raise ValueError('rerun copy hash')
  if os.path.samestat(Path(x['source']['path']).stat(),Path(x['destination']['path']).stat()):raise ValueError('rerun inode alias')
 if e.get('source_dictionary')!=ev(Path(dictionary_source)) or e.get('run_local_dictionary')!=ev(root/'dict'/'mfa_ipa.dict') or e['run_local_dictionary']['path']!=str((root/'dict'/'mfa_ipa.dict').resolve()) or (e['source_dictionary']['size'],e['source_dictionary']['sha256'])!=(e['run_local_dictionary']['size'],e['run_local_dictionary']['sha256']) or os.path.samestat(Path(dictionary_source).stat(),(root/'dict'/'mfa_ipa.dict').stat()):raise ValueError('dictionary evidence')
 if e.get('prepare_manifest_sha256')!=digest(root/'prepare_manifest.json'):raise ValueError('prepare manifest hash')
 return True
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);p.add_argument('--source-dir',type=Path,default=SOURCE);p.add_argument('--dictionary-source',type=Path,default=DEFAULT_DICT);p.add_argument('--asr-python',default=DEFAULT_ASR_PYTHON);p.add_argument('--asr-model',default=DEFAULT_ASR_MODEL);a=p.parse_args()
 try:verify(a.run_root,a.source_dir,a.dictionary_source,a.asr_python,a.asr_model);print('v4 ready evidence verified')
 except Exception as e:print('ERROR: '+str(e),file=sys.stderr);return 1
 return 0
if __name__=='__main__':raise SystemExit(main())
