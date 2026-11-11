import csv,json,time,concurrent.futures,traceback
from pathlib import Path
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
D=Path("/mnt/data/users/zhjia/datasets-extension-20261110/sku110k/raw/SKU110K_fixed")
OUT=Path("/mnt/data/users/zhjia/datasets-extension-20261110/control/sku-full-image-decode-tolerant.json")
images={}
for p in (D/"annotations").glob("annotations_*.csv"):
 for row in csv.reader(p.open()): images[row[0]]=(int(row[6]),int(row[7]))
def check(item):
 name,expected=item
 try:
  with Image.open(D/"images"/name) as im:
   actual=im.size
   assert actual==expected,(actual,expected)
   im.convert("RGB").load()
  return None
 except Exception as e:return {"file":name,"error":repr(e)}
t=time.monotonic();fail=[];done=0
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for r in pool.map(check,images.items()):
  done+=1
  if r:fail.append(r)
  if done%500==0: print("DECODE",done,"/",len(images),"failures",len(fail),flush=True)
report={"status":"PASS" if not fail else "FAILED","scope":"All original split-listed JPEGs decoded via Pillow LOAD_TRUNCATED_IMAGES=True, convert(RGB).load; raw bytes retained, no drop/repair; tolerant decoder can conceal missing entropy data, disclosed", "images":len(images),"decoded":done,"failures":fail,"seconds":time.monotonic()-t}
OUT.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)
