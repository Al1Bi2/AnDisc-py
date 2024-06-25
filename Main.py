import multiprocessing as mp

import wx

import proc
import os
dirname = os.path.dirname(__file__)
maskfilename = os.path.join(dirname, 'no_signal.png')
q_in = mp.Queue()
q_out = mp.Queue()
q_2proc = mp.Queue()

fps = 36

proc_thread = mp.Process(name="Proc", target=proc.processing, args=(q_in, q_out, q_2proc,))
audio_thread = mp.Process(name="audio", target=proc.audio)
virtual_cam_thread = mp.Process(name="virtual_cam", target=proc.virtual_cam, args=(q_out,))


def start_back():
    audio_thread.start()
    virtual_cam_thread.start()
    proc_thread.start()


def stop_back():
    while not q_2proc.empty():
        q_2proc.get()
    q_2proc.close()

    while not q_in.empty():
        q_in.get()
    q_in.close()

    while not q_out.empty():
        q_out.get()
    q_out.close()

    proc_thread.terminate()
    audio_thread.terminate()
    virtual_cam_thread.terminate()

    proc_thread.join()
    audio_thread.join()
    virtual_cam_thread.join()


# ******************************GUI************************************#
class MyFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        # begin wxGlade: MyFrame.__init__
        kwds["style"] = kwds.get("style", 0) | wx.DEFAULT_FRAME_STYLE | wx.MAXIMIZE
        wx.Frame.__init__(self, *args, **kwds)
        self.SetSize((640, 545))
        self.SetTitle("frame")
        self.Layout()
        self.Centre()

        self.cam_on = False
        self.panel_1 = wx.Panel(self, wx.ID_ANY)
        self.panel_1.SetSize((640, 545))
        sizer_2 = wx.BoxSizer(wx.VERTICAL)
        self.panel_1.SetSizer(sizer_2)

        sizer_3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_2.Add(sizer_3, 0, wx.EXPAND, 0)

        self.mask = wx.Button(self.panel_1, wx.ID_ANY, "Open mask")
        sizer_3.Add(self.mask, 0, wx.ALL, 0)

        self.checkbox_1 = wx.CheckBox(self.panel_1, wx.ID_ANY, "No Face", style=wx.ALIGN_RIGHT | wx.CHK_2STATE)
        self.checkbox_1.SetValue(1)
        sizer_3.Add(self.checkbox_1, 0, 0, 0)

        self.checkbox_2 = wx.CheckBox(self.panel_1, wx.ID_ANY, "Mask", style=wx.ALIGN_RIGHT)
        sizer_3.Add(self.checkbox_2, 0, 0, 0)

        self.slider = wx.Slider(self.panel_1, value=100, minValue=80, maxValue=200,
                                style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        sizer_3.Add(self.slider, 0, 0, 0)

        self.displayPanel = wx.Panel(self.panel_1, wx.ID_ANY)  # image panel
        self.displayPanel.SetSize((640, 545))
        self.displayPanel.SetDoubleBuffered(True)
        self.image = wx.Bitmap(maskfilename, wx.BITMAP_TYPE_ANY)
        self.displayPanel.Bind(wx.EVT_PAINT, self.onPaint)
        sizer_2.Add(self.displayPanel, 1, wx.EXPAND, 0)

        self.timex = wx.Timer(self, wx.ID_OK)  # Timer and its event
        self.timex.Start(1000 // fps)
        self.Bind(wx.EVT_TIMER, self.Redraw, self.timex)

        self.Bind(wx.EVT_BUTTON, self.OnOpen, self.mask)  # Other events
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_CHECKBOX, self.Check1Box, self.checkbox_1)
        self.Bind(wx.EVT_CHECKBOX, self.Check2Box, self.checkbox_2)
        self.Bind(wx.EVT_SLIDER, self.OnSliderScroll)

        self.Bind(wx.EVT_CLOSE, self.OnExit)
        # end wxGlade

    def OnSliderScroll(self, e):
        obj = e.GetEventObject()
        val = obj.GetValue()
        q_2proc.put((3, val))

    def onPaint(self, evt):
        dc = wx.BufferedPaintDC(self.displayPanel)
        dc.DrawBitmap(self.image, 0, 0, True)

    def OnEraseBackground(self, evt):
        pass

    def Redraw(self, event):
        #print(self.image.GetSize())
        #print(q_in.empty())

        if self.cam_on:

            tmp = q_in.get()
            #print(tmp)

            self.image.CopyFromBuffer(tmp)
            
            self.Refresh()
        else:
            if not q_in.empty():
                self.cam_on = True
        #print("____")

    def Check1Box(self, event):
        state = self.checkbox_1.GetValue()
        q_2proc.put((1, state))

    def Check2Box(self, event):
        state = self.checkbox_2.GetValue()
        q_2proc.put((2, state))

    def OnOpen(self, event):

        with wx.FileDialog(self, "Open Mask file",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
                           wildcard="PNG and JPEG files (*.png;*.jpeg)|*.png;*.jpeg") as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            pathname = fileDialog.GetPath()
            q_2proc.put((0, pathname))

    def OnExit(self, event):
        self.timex.Stop()
        stop_back()
        self.Destroy()


class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, wx.ID_ANY, "")
        start_back()
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True


# end of GUI classes

if __name__ == '__main__':
    app = MyApp(0)
    app.MainLoop()
