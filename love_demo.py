# bug-free-shipin
#视频爬取vip连接
# 正则表达式
import re
# 桌面软件
import tkinter as tk
# url解析包
from urllib import parse
# 消息盒子包 弹出错误信息
import tkinter.messagebox as msgbox
# 控制浏览器的包
import webbrowser

class App:
    # 1.重写构造方法
    def __init__(self, width=500, height=300):
        # 定义类属性
        self.w = width
        self.h = height

        # 软件名
        self.title = '视频解析助手'
        self.root = tk.Tk(className=self.title)

        # 用户输入的视频地址链接
        self.url = tk.StringVar()

        # 定义播放源 第三方的播放源
        self.v = tk.IntVar()
        self.v.set(1)

        # 软件空间划分
        frame_1 = tk.Frame(self.root)
        frame_2 = tk.Frame(self.root)

        # 控件设置
        group = tk.Label(frame_1, text='播放通道', padx=10, pady=10)
        tb = tk.Radiobutton(frame_1, text='第一通道', variable=self.v, value=1, width=10, height=3)

        label = tk.Label(frame_2, text='请输入视频链接:')
        entry = tk.Entry(frame_2, textvariable=self.url, highlightcolor='Fuchsia', highlightthickness=1, width=35)
        play = tk.Button(frame_2, text='播放', font=('楷体', 12), fg='Purple', width=2, height=1, command=self.video_play)

        # 控件布局
        frame_1.pack()
        frame_2.pack()

        # 确定位置
        group.grid(row=0, column=0)
        tb.grid(row=0, column=1)

        label.grid(row=0, column=0)
        entry.grid(row=0, column=1)
        play.grid(row=0, column=2, ipadx=10, ipady=10)

    # 定义播放功能
    def video_play(self):
        # 定义第三方的视频网站地址
        port = 'http://www.wmxz.wang/video.php?url='

        # 使用正则表达式去判断用户输入的地址是否合法
        if re.match(r'^https?:/{2}\w.+$', self.url.get()):
            # 拿到用户输入的地址
            ip = self.url.get()
            # 网址解析
            ip = parse.quote_plus(ip)
            # 使用浏览器打开刚刚解析完成的地址
            webbrowser.open(port + ip)
        else:
            msgbox.showerror(title='错误', message='视频链接地址无效, 请重新输入....')


    # 启动软件的函数
    def loop(self):
        self.root.resizable(True, True)
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.loop()
