import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv 
import warnings
warnings.filterwarnings('ignore')
from pyAudioAnalysis import ShortTermFeatures
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from bioinfokit.analys import stat
import pylab

for j in os.listdir("dataset/IRMAS_Training_Data"):
    break
    print(j)
    if j != '.DS_Store':
        ind=1
        for i in os.listdir("dataset/IRMAS_Training_Data/"+j):
            # print(i)
            instrument = pd.DataFrame(columns=["br1","br2", "br3", "spectral_centroid",'spectral_spread', 'spectral_entropy', 'spectral_flux','spectral_rolloff'])

            x, Fs = librosa.load("dataset/IRMAS_Training_Data/"+j+"/"+ str(i))
            # print(x, len(x), Fs)
            F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
            instrument["spectral_centroid"] = F[3]
            instrument["spectral_spread"] = F[4]
            instrument["spectral_entropy"] = F[5]
            instrument["spectral_flux"] = F[6]
            instrument["spectral_rolloff"] = F[7]

            if len(i.split("[")) == 4:
                instrument["br1"] = i.split("[")[1][0:3]
                instrument["br2"] = i.split("[")[2][0:3]
                instrument["br3"] = i.split("[")[3][0:3]
            if len(i.split("[")) == 3:
                instrument["br1"] = i.split("[")[1][0:3]
                instrument["br3"] = i.split("[")[2][0:3]

            if ind == 1:
                data_instrument=instrument
                ind=2
            if ind==2:
                data_instrument = pd.concat([data_instrument, instrument])
                
        data_instrument.reset_index(inplace=True)
        data_instrument.to_csv("Spectral_"+j+".csv") 

data_voi = pd.read_csv("data/Spectral_voi.csv")
data_voi[["br1", "br2", "br3"]].drop_duplicates()

k = 1
for inst in ["voi", "cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio"]:
    d = pd.read_csv("data/Spectral_"+inst+".csv")
    if k ==1:
        testData=d
        k=2
    if k==2:
        testData = pd.concat([testData, d])

testData.reset_index(inplace=True)

# for y in ['spectral_centroid','spectral_spread', 'spectral_entropy', 'spectral_flux', 'spectral_rolloff']:
#     sns.boxplot(x='br1', y=y, data=testData, notch=True, showcaps=False,
#         flierprops={"marker": "x"},
#         boxprops={"facecolor": (.4, .6, .8, .5)},
#         medianprops={"color": "coral"})
#     plt.savefig(f'{y}.png')
#     plt.show()

for feature in ["spectral_centroid", "spectral_spread","spectral_entropy", "spectral_flux","spectral_rolloff"]:
    print(feature)
    inst = ["voi", "cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio"]
    fvalue, pvalue = stats.f_oneway(testData.loc[testData.br1==inst[0], feature], 
                                    testData.loc[testData.br1==inst[1], feature], 
                                    testData.loc[testData.br1==inst[2], feature], 
                                    testData.loc[testData.br1==inst[3], feature],
                                    testData.loc[testData.br1==inst[4], feature],
                                    testData.loc[testData.br1==inst[5], feature],
                                    testData.loc[testData.br1==inst[6], feature],
                                    testData.loc[testData.br1==inst[7], feature],
                                    testData.loc[testData.br1==inst[8], feature],
                                    testData.loc[testData.br1==inst[9], feature],
                                    testData.loc[testData.br1==inst[10], feature]
                                    )

    # get ANOVA table as R like output
    model = ols(feature+' ~ C(br1)', data=testData).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    plt.rc('figure', figsize=(5, 1))
    plt.text(0.01, 0.05, str(anova_table), {'fontsize': 9}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'data/images/stat_tables/{feature}_anova.png')
    plt.close()
    # perform multiple pairwise comparison (Tukey's HSD)
    # unequal sample size data, tukey_hsd uses Tukey-Kramer test
    res = stat()
    res.tukey_hsd(df=testData, res_var=feature, xfac_var='br1', anova_model=feature+' ~ C(br1)')
    plt.rc('figure', figsize=(6, 11))
    plt.text(0.01, 0.05, str(res.tukey_summary), {'fontsize': 9}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'data/images/stat_tables/{feature}_tukey.png')
    plt.close()
quit()
f, axs=plt.subplots(1,1, figsize=(10,7))
sns.histplot(data=testData[(testData.br1=="cel") & (testData.br2=="nod") & (testData.br3=="cla")], x="spectral_centroid", color="red", ax=axs, bins=np.arange(0, 0.5, 0.01), label="glas")
sns.histplot(data=testData[(testData.br1=="cla") & (testData.br2=="nod") & (testData.br3=="cla")], x="spectral_centroid", color="blue",ax=axs, bins=np.arange(0, 0.5, 0.01), label="akustična gitara")
sns.histplot(data=testData[(testData.br1=="pia") & (testData.br2=="nod") & (testData.br3=="cla")], x="spectral_centroid", color="green",ax=axs, bins=np.arange(0, 0.5, 0.01), label="električna gitara")
sns.histplot(data=testData[(testData.br1=="vio") & (testData.br2=="nod") & (testData.br3=="cla")], x="spectral_centroid", color="purple",ax=axs, bins=np.arange(0, 0.5, 0.01), label="klavir")
axs.legend()
# plt.savefig('data/images/spectral_centroid_hisplot.png')
plt.show()
print(np.var(testData[(testData.br1=="cel") & (testData.br2=="nod") & (testData.br3=="cla")].spectral_centroid))
print(np.var(testData[(testData.br1=="cla") & (testData.br2=="nod") & (testData.br3=="cla")].spectral_centroid))
print(np.var(testData[(testData.br1=="pia") & (testData.br2=="nod") & (testData.br3=="cla")].spectral_centroid))
print(np.var(testData[(testData.br1=="vio") & (testData.br2=="nod") & (testData.br3=="cla")].spectral_centroid))

testData = testData[(testData.br1.isin(["cel","cla","pia","vio"])) & (testData.br2=="nod") & (testData.br3=="cla")]
stats.probplot(testData[testData.br1=="cel"]['spectral_centroid'], dist="norm", plot=pylab)
pylab.savefig('data/images/normplot/cel_sc.png')
pylab.show()
stats.probplot(testData[testData.br1=="cla"]['spectral_centroid'], dist="norm", plot=pylab)
pylab.savefig('data/images/normplot/cla_sc.png')
pylab.show()
stats.probplot(testData[testData.br1=="pia"]['spectral_centroid'], dist="norm", plot=pylab)
pylab.savefig('data/images/normplot/pia_sc.png')
pylab.show()
stats.probplot(testData[testData.br1=="vio"]['spectral_centroid'], dist="norm", plot=pylab)
pylab.savefig('data/images/normplot/vio_sc.png')
pylab.show()

model = ols('spectral_centroid ~ C(br1)', data=testData).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

res = stat()
res.tukey_hsd(df=testData, res_var='spectral_centroid', xfac_var='br1', anova_model='spectral_centroid ~ C(br1)')
print(res.tukey_summary)