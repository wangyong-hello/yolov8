#######根据标签文件统计目标种类以及数量#  *coding:utf-8*
#跑xml标签需要修改：label_path，RESULTS_FILES_PATH，将XML设置为True
#跑yolo标签需要修改：label_path，RESULTS_FILES_PATH，将XML设置为False,names（添加类名列表）
import imp
import os
from turtle import left
import xml.etree.ElementTree as ET
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import operator
 
def parse_obj(label_path, filename):
  tree=ET.parse(label_path+filename)
  objects=[]
  for obj in tree.findall('object'):
    obj_struct={}
    obj_struct['name']=obj.find('name').text
    objects.append(obj_struct)
  return objects

def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])

def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    # 
    if true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - orange -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        # plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        # plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')    #plt.barh是水平制直方图而plt.bar垂直制直方图
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        # plt.barh(range(n_classes), sorted_values, color=plot_color) #plt.barh是水平制直方图而plt.bar垂直制直方图
        plt.barh(range(n_classes), sorted_values, color=plot_color) #plt.barh是水平制直方图而plt.bar垂直制直方图
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val) # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    # fig.canvas.set_window_title(window_title)
    fig.canvas.manager.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height 
    top_margin = 0.15 # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the ficcgure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()

def main(RESULTS_FILES_PATH,label_path,XML,names=None):
    total=0  #所有框个数    
###################### 统计每个类及框的个数 ###############################
    if XML:
        filenamess=os.listdir(label_path)
        filenames=[]
        for name in filenamess:
            if name.endswith('xml'):
                name=name.replace('.xml','')
                filenames.append(name)
        recs={}   #是一个字典，其中一个元素000002为：[{'name': 'leaf'}, {'name': 'can'}, {'name': 'branch'}]
        names=[]
        num_objs={} #每个类的框的个数。是一个字典。

        for i,name in enumerate(filenames):
            recs[name]=parse_obj(label_path, name+ '.xml' )
        for name in filenames:
            # print(name)
            for object in recs[name]:
                if object['name'] not in num_objs.keys():
                    num_objs[object['name']]=1
                else:
                    num_objs[object['name']]+=1
                if object['name'] not in names:
                    names.append(object['name'])
        for name in names:
            print('{}:{}个'.format(name,num_objs[name]))
        for i in num_objs.values():
            total += i
        print("=====>total number",total)
    else:       
        class_num = len(names) # 样本类别数
        class_list = [i for i in range(class_num)]
        class_num_list = [0 for i in range(class_num)]
        labels_list = os.listdir(label_path)
        num_objs={}
        # num_objs
        for i in labels_list:
            file_path = os.path.join(label_path, i)
            file = open(file_path, 'r')  # 打开文件
            file_data = file.readlines()  # 读取所有行
            for every_row in file_data:
                class_val = every_row.split(' ')[0]
                class_ind = class_list.index(int(class_val))
                class_num_list[class_ind] += 1
            file.close()
        # 输出每一类的数量以及总数
        for i,name in enumerate(names):
            # print(name+':'+str(class_num_list[i]))
            num_objs[name]=class_num_list[i]
            # num_objs.update(name:class_num_list[i])
        print(num_objs)
        for i in num_objs.values():
            total += i
        print("=====>total number",total)

####################################  绘图   ##################################### 
    gt_counter_per_class=num_objs
    window_title = "ground-truth-info"
    plot_title = "test_ground_truth\n"
    plot_title += "("+str(len(names)) + "classes    " +"total_number:"+str(total)+  ")" 
    x_label = "Number of objects per class"
    output_path = RESULTS_FILES_PATH + "/test.png"
    to_show = True   #是否显示图形
    # plot_color = 'forestgreen'
    plot_color = 'blue'
    draw_plot_func(gt_counter_per_class,len(names),window_title,plot_title,x_label,
        output_path,to_show,plot_color,'',)

if __name__ == '__main__':
    label_path=r'C:\Users\14833\Desktop\yolov5-6.1_official\datasets\garbage\test\labels'+'\\'  #标签所存放的路径
    RESULTS_FILES_PATH=r'C:\Users\14833\Desktop\yolov5-6.1_official\datasets' 
    XML=False  #默认读取的标签是xml格式 
    names=['leaf','branch','bottle','grass','milk_box','plastic_bag',    #如果是yolo格式的需要修改这个列表
         'can','plastic_box','glass_bottle','bad_fruit','ball']  
    main(RESULTS_FILES_PATH,label_path,XML,names) 
