import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle, os
from jiscode import jiscode

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

# home = os.path.expanduser('~') + '/ETL/ETL9B'
home = 'E:/DATA/ETL/ETL9B'
# 保存先や画像サイズの指定 --- (*1)
out_dir =home + "/png-etl9b" # 画像データがあるディレクトリ
im_size = 32 # 画像サイズ
# save_file = home + "/ETL9B_28.pickle" # 保存先
save_file = home + "/ETL9B_{}_gaus.pickle".format(im_size) # 保存先
jis_code_file = home + '/ETL9BJISCODE.picle'
plt.figure(figsize=(9, 10)) # 出力画像を大きくする
jis1 = jiscode()

files = glob.glob(out_dir +"/*")
kanji = []
for f in files:
    kanji.append(int(os.path.basename(f)[0:4],16))
# カタカナの画像が入っているディレクトリから画像を取得 --- (*2)
# kanji = list(range(177, 220 + 1))
# kanji.append(166) # ヲ
# kanji.append(221) # ン
result = []
k=0
out_size = 0
code_mat = []
for i, code in enumerate(kanji):
    code_uni = jis1.jis2uni(code)
    # img_dir = out_dir + "/" + "{0:02X}({1:})".format(code, chr(code_uni))
    img_dir = out_dir + "/" + "{0:02X}".format(code)

    # img_dir = out_dir + "/" + str(code)
    fs = glob.glob(img_dir + "/*")
    print("dir=",  img_dir)
    # 画像を読み込んでグレイスケールに変換しリサイズする --- (*3)
    out_size += 1
    code_mat.append(code)
    for j, f in enumerate(fs):
        try:
            f = f.replace('\\', '/')
            img = imread(f)
            s = img.shape
            if s[0]>0 and s[1]>0:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_gray = dst = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=0)
                # plt.imshow(img_gray)
                # plt.show()
                img = cv2.resize(img_gray, (im_size, im_size))
                result.append([i, img])                # Jupyter Notebookで画像を出力
                k += 1
                if k<=50 :
                    plt.subplot(10, 5, k)
                    plt.axis("off")
                    plt.title(str(i))
                    plt.imshow(img, cmap='gray')
        except :
            pass
# ラベルと画像のデータを保存 --- (*4)

pickle.dump((im_size, out_size, result), open(save_file, "wb"))
pickle.dump((code_mat), open(jis_code_file, "wb"))
plt.show()
print("ok")
