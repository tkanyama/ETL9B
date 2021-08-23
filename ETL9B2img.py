# ETL1Cのファイルを読み込む
import struct
from PIL import Image, ImageEnhance
import glob, os
from jiscode import jiscode

# 出力ディレクトリ
# home = os.path.expanduser('~') + '/ETL/ETL9B'
home = 'E:/DATA/ETL/ETL9B'
outdir = home + "/png-etl9b/"
if not os.path.exists(outdir): os.mkdir(outdir)
jis1 = jiscode()
# ETL1ディレクトリ以下のファイルを処理する --- (*1)
files = glob.glob(home + "/ETL9B/*")
for fname in files:
    fname = fname.replace('\\', '/')
    if fname == home + "/ETL9B/ETL9INFO": continue # 情報ファイルは飛ばす
    if fname == home + "/png-etl9b": continue # 情報ファイルは飛ばす
    print(fname)
    # ETL1のデータファイルを開く --- (*2)
    f = open(fname, 'rb')
    f.seek(0)
    i = 0
    while True:
        # メタデータ＋画像データの組を一つずつ読む --- (*3)
        s = f.read(576)
        if not s: break

        # バイナリデータなのでPythonが理解できるように抽出 --- (*4)
        # r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
        r = struct.unpack('>2H4s504s64x', s)
        if r[0]>0:
            code_ascii = r[1]
            code_jis = r[1]
            code_uni = jis1.jis2uni(code_jis)
            # if (i+1) % 80 ==0:
            #     print(chr(jis1.jis2uni(code_jis)))
            # else:
            #     print(chr(jis1.jis2uni(code_jis)), end="")
            # 画像データとして取り出す --- (*5)
            iF = Image.frombytes('1', (64, 63), r[3], 'raw')
            iP = iF.convert('L')
            # 画像を鮮明にして保存
            # dir = outdir + "/" + "{0:02X}_{1:}".format(code_jis, chr(code_uni))
            dir = outdir + "/" + "{0:02X}".format(code_jis)
            if not os.path.exists(dir): os.mkdir(dir)
            fn = "{0:02X}-{1:02X}{2:04X}.png".format(code_jis, r[0], i)
            fullpath = dir + "/" + fn
            #if os.path.exists(fullpath): continue
            enhancer = ImageEnhance.Brightness(iP)
            iE = enhancer.enhance(16)
            iE.save(fullpath, 'PNG')
        i += 1
print("ok")
