
def ShowChinese():
    from matplotlib.pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
    mpl.rcParams['axes.unicode_minus'] = False  # 负号显示
