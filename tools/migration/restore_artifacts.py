#!/usr/bin/env python3
"""Public, checksum-verified selective artifact recovery; Windows/Linux/macOS."""
import argparse,collections,hashlib,json,os,shutil,tarfile,tempfile,time,urllib.parse,urllib.request
from pathlib import Path,PurePosixPath
REPO='LamTNguyen/TQK-research-artifacts'
CAT='migration_windows_20261022/MIGRATION_CATALOG.json'
LEGACY='STORAGE_MANIFEST.json'
SSD='ssd_completion_20261022/SSD_CATALOG.json'
BASE='https://huggingface.co/datasets/'+REPO+'/resolve/'
def url(path,revision='main'):
 return BASE+urllib.parse.quote(revision,safe='')+'/'+urllib.parse.quote(path,safe='/')
def read_json(path):
 with urllib.request.urlopen(url(path)+'?download=true&catalog_refresh='+str(time.time_ns()),timeout=90) as f:return json.load(f)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def verify(path,record):
 if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise RuntimeError('Checksum mismatch: '+str(path))
def download(record,path):
 with urllib.request.urlopen(url(record['path'],record['revision']),timeout=120) as src,path.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024**2)
 verify(path,record)
def safe_target(root,name):
 parts=PurePosixPath(name).parts
 if not parts or PurePosixPath(name).is_absolute() or any(x in ('..','.') or ':' in x or '\\' in x for x in parts):raise ValueError('Unsafe artifact path')
 root=root.resolve();p=root.joinpath(*parts)
 if p.resolve()!=root and root not in p.resolve().parents:raise ValueError('Destination escapes workspace')
 return p
def install_stream(src,target,record):
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():verify(target,record);return
 fd,temp=tempfile.mkstemp(prefix='.tqk-part-',dir=target.parent);tmp=Path(temp)
 try:
  with os.fdopen(fd,'wb') as out:shutil.copyfileobj(src,out,8*1024**2)
  verify(tmp,record);os.replace(tmp,target)
 finally:tmp.unlink(missing_ok=True)
def unpack_selected(archive,members,workspace):
 import zstandard
 remaining=dict(members)
 with archive.open('rb') as compressed,zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(compressed) as stream,tarfile.open(fileobj=stream,mode='r|') as tar:
  for member in tar:
   if member.name not in remaining:continue
   aliases=remaining.pop(member.name)
   if isinstance(aliases,tuple):aliases=[aliases] # Compatibility with earlier callers.
   name,record=aliases[0]
   if not member.isfile() or member.size!=record['bytes']:raise RuntimeError('Unexpected archive member')
   target=safe_target(workspace,name)
   with tar.extractfile(member) as src:install_stream(src,target,record)
   print('RESTORED',name)
   for other,expected in aliases[1:]:
    if expected['sha256']!=record['sha256'] or expected['bytes']!=record['bytes']:raise RuntimeError('Invalid deduplicated alias')
    with target.open('rb') as src:install_stream(src,safe_target(workspace,other),expected)
    print('RESTORED',other)
  if remaining:raise RuntimeError('Requested members missing: '+str(list(remaining)))
def load_entries(current,legacy,ssd=None):
 entries={};archives={}
 for name,r in legacy.get('files',{}).items():
  if r.get('storage_archive'):
   arname=r['storage_archive'];ar=legacy['files'][arname];archives[arname]=dict(ar,path=arname)
   entries[name]=dict(r,archive=arname,member=r['archive_member'])
  elif not r.get('archive'):
   entries[name]=dict(r,direct=dict(r,path=name))
  else:archives[name]=dict(r,path=name)
 archives.update(current.get('archives',{}));entries.update(current.get('files',{}))
 if ssd:
  if ssd.get('status')!='COMPLETE':raise RuntimeError('SSD backup is not complete')
  archives.update(ssd.get('archives',{}));entries.update(ssd.get('files',{}))
 return entries,archives
def restore(entries,archives,names,workspace):
 groups=collections.defaultdict(lambda:collections.defaultdict(list));direct=[]
 for name in names:
  r=entries[name];target=safe_target(workspace,name)
  if target.exists():verify(target,r);print('ALREADY_VERIFIED',name);continue
  if r.get('archive'):groups[r['archive']][r['member']].append((name,r))
  else:direct.append((name,r))
 with tempfile.TemporaryDirectory(prefix='tqk-restore-') as tmp:
  tmp=Path(tmp)
  for name,r in direct:
   p=tmp/'download';download(r['direct'],p)
   with p.open('rb') as f:install_stream(f,safe_target(workspace,name),r)
   p.unlink();print('RESTORED',name)
  for ar,members in groups.items():
   p=tmp/'archive.tar.zst';print('DOWNLOAD_ARCHIVE',ar,archives[ar]['bytes']);download(archives[ar],p)
   unpack_selected(p,members,workspace);p.unlink()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['list','restore']);p.add_argument('--prefix',default='');p.add_argument('--path',action='append',default=[]);p.add_argument('--workspace',type=Path,default=Path.cwd());p.add_argument('--catalog',type=Path);p.add_argument('--legacy-catalog',type=Path);p.add_argument('--ssd-catalog',type=Path)
 a=p.parse_args()
 ssd_only=a.prefix.startswith('ssd/') or (bool(a.path) and all(n.startswith('ssd/') for n in a.path))
 if ssd_only:
  ssd=json.loads(a.ssd_catalog.read_text()) if a.ssd_catalog else read_json(SSD)
  entries,archives=load_entries({}, {}, ssd)
 else:
  current=json.loads(a.catalog.read_text()) if a.catalog else read_json(CAT);legacy=json.loads(a.legacy_catalog.read_text()) if a.legacy_catalog else read_json(LEGACY)
  ssd=json.loads(a.ssd_catalog.read_text()) if a.ssd_catalog else None
  entries,archives=load_entries(current,legacy,ssd)
  if not a.path and not a.prefix:print('SSD results: rerun with --prefix ssd/ (separate self-contained index)')
 names=a.path or sorted(n for n in entries if n.startswith(a.prefix))
 for name in names:
  if name not in entries:p.error('Unknown artifact: '+name)
 if a.action=='list':
  for n in names:print(entries[n]['bytes'],n)
  print('FILES',len(names),'UNCOMPRESSED_BYTES',sum(entries[n]['bytes'] for n in names));return
 if not a.path and not a.prefix:p.error('Choose --path or --prefix; bulk recovery is never implicit')
 restore(entries,archives,names,a.workspace)
if __name__=='__main__':main()
