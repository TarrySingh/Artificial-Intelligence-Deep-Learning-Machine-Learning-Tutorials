# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

from tkinter import ttk
import tkinter as tk
import os
from PIL import Image, ImageTk
from data_utils import *
from config import *
import numpy as np
import enum


review_img_width = 3000
review_img_height = 1500


class Criteria(enum.Enum):
    # too short, many_people, skeleton_back, skeleton_missing, skeleton_side, skeleton_small,is_picture
    too_short = 0
    many_people = 1
    skeleton_back = 2
    skeleton_missing = 3
    skeleton_side = 4
    skeleton_small = 5
    is_picture = 6


class ReviewApp:
    MODE = 'ALL'
    vid = '-1'

    def __init__(self):
        self.win = tk.Tk()
        self.win.geometry("1500x800+100+100")

        self.make_frame()
        self.make_label()
        self.make_filtering_box()
        self.make_img_canvas()
        self.make_view_combobox()

        self.make_vid_treeView()
        self.vid_tree.bind("<Double-1>", self.OnVideoListClick)

        self.make_clip_treeView()
        self.clip_tree.bind("<Double-1>", self.OnClipListClick)
        self.clip_tree.bind("<<TreeviewSelect>>", self.OnClipTreeSelect)
        self.img_canvas.focus_set()

        self.win.mainloop()

    def make_frame(self):
        # main grid
        self.win.rowconfigure(0, weight=1)
        self.win.rowconfigure(1, weight=9)
        self.win.columnconfigure(0, weight=1)

        self.top_frame = tk.Frame(self.win, bg='#e9e9e9')
        self.top_frame.grid(row=0, sticky='nsew')
        self.top_frame.columnconfigure(0, weight=1)
        self.top_frame.columnconfigure(1, weight=12)

        self.top_frame.rowconfigure(0, weight=1)
        self.top_frame.rowconfigure(1, weight=1)
        self.top_frame.rowconfigure(2, weight=1)

        self.bottom_frame = tk.Frame(self.win)
        self.bottom_frame.grid(row=1, sticky='nsew', padx=5, pady=5)
        self.bottom_frame.columnconfigure(0, weight=1)
        self.bottom_frame.columnconfigure(1, weight=1)

        # bottom frame grid
        self.bottom_frame.columnconfigure(0, weight=1)
        self.bottom_frame.columnconfigure(1, weight=1)
        self.bottom_frame.columnconfigure(2, weight=15)
        self.bottom_frame.rowconfigure(0, weight=1)

        self.img_frame = tk.Frame(self.bottom_frame)
        self.img_frame.grid(row=0, column=2, sticky='nsew', padx=5, pady=5)

    def make_label(self):
        self.tx_vid_name = tk.Label(self.top_frame, bg='#8C8C8C', text='No selected video')
        self.tx_clip_interval = tk.Label(self.top_frame, bg='#8C8C8C', text='No selected clip')
        self.tx_vid_name.grid(row=0, column=0, sticky=(tk.N + tk.S + tk.E + tk.W))
        self.tx_clip_interval.grid(row=1, column=0, sticky=(tk.N + tk.S + tk.E + tk.W))

    def make_view_combobox(self):
        self.mode = tk.StringVar()
        self.view_combo = ttk.Combobox(self.top_frame, values=('ALL', 'TRUE', 'FALSE'), textvariable=self.mode)
        self.view_combo.grid(row=2, column=0, sticky=(tk.N + tk.S + tk.E + tk.W), padx=5, pady=5)
        self.view_combo.current(0)
        self.view_combo.bind('<<ComboboxSelected>>', self.OnComboSelected)

    def make_filtering_box(self):
        self.skeltonoptionFrame = tk.Frame(self.top_frame, bg='#e9e9e9')
        self.skeltonoptionFrame.grid(row=0, column=1, sticky='nsew')
        ratioFrame = tk.Frame(self.top_frame, bg='#e9e9e9')
        ratioFrame.grid(row=1, column=1, sticky='nsew')

        msgFrame = tk.Frame(self.top_frame, bg='#e9e9e9')
        msgFrame.grid(row=2, column=1, sticky='nsew')

        tx_back = tk.Label(ratioFrame, text="looking behind ratio: ", foreground='#3985F8', bg='#e9e9e9')
        tx_back.pack(side=tk.LEFT, padx=5)
        self.tx_ratio_back = tk.Label(ratioFrame, text="None", bg='#e9e9e9')
        self.tx_ratio_back.pack(side=tk.LEFT)

        tx_missing = tk.Label(ratioFrame, text="missing joints ratio: ", foreground='#3985F8', bg='#e9e9e9')
        tx_missing.pack(side=tk.LEFT, padx=10)
        self.tx_ratio_missing = tk.Label(ratioFrame, text="None", bg='#e9e9e9')
        self.tx_ratio_missing.pack(side=tk.LEFT)

        tx_side = tk.Label(ratioFrame, text="looking sideways ratio: ", foreground='#3985F8', bg='#e9e9e9')
        tx_side.pack(side=tk.LEFT, padx=10)
        self.tx_ratio_side = tk.Label(ratioFrame, text="None", bg='#e9e9e9')
        self.tx_ratio_side.pack(side=tk.LEFT)

        tx_small = tk.Label(ratioFrame, text="small person ratio: ", foreground='#3985F8', bg='#e9e9e9')
        tx_small.pack(side=tk.LEFT, padx=10)
        self.tx_ratio_small = tk.Label(ratioFrame, text="None", bg='#e9e9e9')
        self.tx_ratio_small.pack(side=tk.LEFT)

        tx_diff = tk.Label(ratioFrame, text="frame diff: ", foreground='#3985F8', bg='#e9e9e9')
        tx_diff.pack(side=tk.LEFT, padx=10)
        self.tx_frame_diff = tk.Label(ratioFrame, text="None", bg='#e9e9e9')
        self.tx_frame_diff.pack(side=tk.LEFT)

        tx_option = tk.Label(self.skeltonoptionFrame, text='Criteria: ', foreground='#3985F8', bg='#e9e9e9')
        tx_option.pack(side=tk.LEFT, padx=5, pady=5)
        tx_res = tk.Label(msgFrame, text='Message:', foreground='#3985F8', bg='#e9e9e9')
        tx_res.pack(side=tk.LEFT, padx=5)
        self.message = tk.Label(msgFrame, text=' ', bg='#e9e9e9')
        self.message.pack(side=tk.LEFT)

        skeleton_option = ["too Short", "many people", "looking behind", "joint missing", "sideways", "small", "picture"]
        self.item = []
        for i in range(7):
            self.item.append(tk.IntVar())

        for val, option in enumerate(skeleton_option):
            tk.Checkbutton(self.skeltonoptionFrame,
                           text=option,
                           padx=5,
                           pady=5,
                           bg='#e9e9e9',
                           variable=self.item[val],
                           activebackground="blue").pack(side=tk.LEFT, padx=5, pady=5)

    def make_vid_treeView(self):
        self.vid_tree = tk.ttk.Treeview(self.bottom_frame)
        self.vid_tree.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self.vid_tree.heading("#0", text="Video List")

        for file in sorted(glob.glob(VIDEO_PATH + "/*.mp4"), key=os.path.getmtime):
            vid = os.path.split(file)[1][-15:-4]
            self.vid_tree.insert('', 'end', text=vid, values=vid, iid=vid)

    def make_clip_treeView(self):
        self.clip_tree = tk.ttk.Treeview(self.bottom_frame)
        self.clip_tree.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        self.clip_tree.heading("#0", text="Clip List")
        self.clip_tree.tag_configure('False', background='#E8E8E8')

    def make_img_canvas(self):
        self.img_canvas = tk.Canvas(self.img_frame, bg='black')
        self.img_canvas.config(scrollregion=(0, 0, review_img_width, review_img_height))

        hbar = tk.Scrollbar(self.img_frame, orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        hbar.config(command=self.img_canvas.xview)
        vbar = tk.Scrollbar(self.img_frame, orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        vbar.config(command=self.img_canvas.yview)
        self.img_canvas.bind("<MouseWheel>", self._on_mousewheel)

        self.img_canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.img_canvas.pack(expand=tk.YES, fill=tk.BOTH)

    def _on_mousewheel(self, event):
        self.img_canvas.yview_scroll(-1 * event.delta, "units")

    def OnComboSelected(self, event):
        change_mode = self.view_combo.get()

        if change_mode != self.MODE:
            self.MODE = change_mode
            self.load_clip()

    def OnVideoListClick(self, event):
        """ load clip data """
        item = self.vid_tree.identify('item', event.x, event.y)
        vid = self.vid_tree.item(item, "text")
        self.vid = vid

        self.tx_vid_name.configure(text=vid)
        self.tx_clip_interval.configure(text='No selected clip')
        self.img_canvas.delete(tk.ALL)
        self.message.config(text=' ')
        self.tx_ratio_small.config(text='None')
        self.tx_ratio_side.config(text='None')
        self.tx_ratio_missing.config(text='None')
        self.tx_ratio_back.config(text='None')
        self.tx_frame_diff.config(text='None')

        print(vid)

        self.clip_data = load_clip_data(vid)
        self.skeleton = SkeletonWrapper(SKELETON_PATH, vid)
        self.video_wrapper = read_video(VIDEO_PATH, vid)
        self.clip_filter_data = load_clip_filtering_aux_info(vid)

        self.load_clip()
        self.win.update()

    def OnClipListClick(self, event):
        item = self.clip_tree.identify('item', event.x, event.y)
        item_index = int(self.clip_tree.item(item, "values")[0])
        print(item_index, 'Double_Click')

    def OnClipTreeSelect(self, event):
        item = self.clip_tree.item(self.clip_tree.focus())
        item_index = int(self.clip_tree.item(self.clip_tree.focus(), 'values')[0])
        print('Load clip, idx:', item_index)

        # load image
        self.review_clip = self.clip_data[item_index]
        start_frame_no = self.review_clip['clip_info'][0]
        end_frame_no = self.review_clip['clip_info'][1]
        correct_clip = self.review_clip['clip_info'][2]

        image = self.show_clips(clip=self.review_clip, correct_clip=correct_clip)

        b, g, r = cv2.split(image)
        img = cv2.merge((r, g, b))
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)

        self.image = imgtk
        self.img_canvas.delete(tk.ALL)
        self.img_canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)

        # self.img_label.image =  self.image
        # self.img_label.config(image=self.image)
        # self.img_label.place(x=0, y=0)

        # load filtering results
        clip_filter_data = self.clip_filter_data[item_index]
        filtering_results = clip_filter_data['filtering_results']
        message = clip_filter_data['message']
        debugging_info = clip_filter_data['debugging_info']

        # tooshort, many_people, skeleton_back, skeleton_missing, skeleton_side, skeleton_small, is_picture 순서
        self.item[Criteria.too_short.value].set(filtering_results[Criteria.too_short.value])
        self.item[Criteria.many_people.value].set(filtering_results[Criteria.many_people.value])
        self.item[Criteria.skeleton_back.value].set(filtering_results[Criteria.skeleton_back.value])
        self.item[Criteria.skeleton_missing.value].set(filtering_results[Criteria.skeleton_missing.value])
        self.item[Criteria.skeleton_side.value].set(filtering_results[Criteria.skeleton_side.value])
        self.item[Criteria.skeleton_small.value].set(filtering_results[Criteria.skeleton_small.value])
        self.item[Criteria.is_picture.value].set(filtering_results[Criteria.is_picture.value])
        self.message.config(text=message)

        self.tx_ratio_back.config(text=debugging_info[0])
        self.tx_ratio_missing.config(text=debugging_info[1])
        self.tx_ratio_side.config(text=debugging_info[2])
        self.tx_ratio_small.config(text=debugging_info[3])
        self.tx_frame_diff.config(text=debugging_info[4])

        self.tx_clip_interval.configure(text=str(start_frame_no) + ' ~ ' + str(end_frame_no) + '  ' + str(correct_clip))
        # self.win.update()

    def load_clip(self):
        if self.vid == '-1':
            print('Error: load video first')
            return

        # init clip tree
        for i in self.clip_tree.get_children():
            self.clip_tree.delete(i)

        self.tx_clip_interval.configure(text='No selected clip')
        self.img_canvas.delete(tk.ALL)

        for item in self.item:
            item.set(False)

        if self.clip_data and self.skeleton.skeletons != []:
            # load clips
            for i, clip in enumerate(self.clip_data):
                start_frame_no = clip['clip_info'][0]
                end_frame_no = clip['clip_info'][1]
                correct_clip = clip['clip_info'][2]

                if self.MODE == 'ALL':
                    self.clip_tree.insert('', 'end', text=str(start_frame_no) + ' ~ ' + str(end_frame_no), values=i,
                                          iid=i, tag=str(correct_clip))
                elif self.MODE == 'TRUE':
                    if correct_clip:
                        self.clip_tree.insert('', 'end', text=str(start_frame_no) + ' ~ ' + str(end_frame_no), values=i,
                                              iid=i, tag=str(correct_clip))
                elif self.MODE == 'FALSE':
                    if not correct_clip:
                        self.clip_tree.insert('', 'end', text=str(start_frame_no) + ' ~ ' + str(end_frame_no), values=i,
                                              iid=i, tag=str(correct_clip))
        else:
            print('[Error] Data file does not exist')
            self.tx_clip_interval.configure(text="Data file does not exist")

        self.win.update()

    def show_clips(self, clip, correct_clip):
        N_IMAGES_PER_VIEW = 20

        start_frame_no = clip['clip_info'][0]
        end_frame_no = clip['clip_info'][1]
        print(start_frame_no, end_frame_no)  # start and end frame no

        # get frames
        resized_frames = []
        skip_amount = int(max((end_frame_no - start_frame_no) / N_IMAGES_PER_VIEW, 1))
        self.video_wrapper.set_current_frame(start_frame_no)
        skeleton_chunk = self.skeleton.get(start_frame_no, end_frame_no)
        for i in range(end_frame_no - start_frame_no):
            ret, frame = self.video_wrapper.video.read()

            if i % skip_amount == 0:
                # overlay raw skeleton on the frame
                if skeleton_chunk and skeleton_chunk[i]:
                    for person in skeleton_chunk[i]:
                        body_pose = get_skeleton_from_frame(person)
                        frame = draw_skeleton_on_image(frame, body_pose, thickness=5)

                if correct_clip and clip['frames']:
                    # overlay selected skeleton

                    if clip['frames'][i]:
                        body_pose = get_skeleton_from_frame(clip['frames'][i])
                        frame = draw_skeleton_on_image(frame, body_pose, thickness=20)

                resized_frame = cv2.resize(frame, (0, 0), None, .35, .35)
                resized_frames.append(resized_frame)

        # make summary img
        n_imgs_per_row = 4
        n_rows_per_page = 5
        frame_idx = 0
        page_img = []
        for row_idx in range(n_rows_per_page):
            row_img = []
            for col_idx in range(n_imgs_per_row):
                if frame_idx >= len(resized_frames):
                    break

                if row_img == []:
                    row_img = resized_frames[frame_idx]
                else:
                    row_img = np.hstack((row_img, resized_frames[frame_idx]))
                frame_idx += 1

            if page_img == []:
                page_img = row_img
            elif row_img != []:
                n_pad = page_img.shape[1] - row_img.shape[1]
                if n_pad > 0:
                    row_img = np.pad(row_img, ((0, 0), (0, n_pad), (0, 0)), mode='constant')
                page_img = np.vstack((page_img, row_img))

        return page_img


if __name__ == '__main__':
    myReviewApp = ReviewApp()
