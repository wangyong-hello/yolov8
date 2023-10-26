####################将xml标签转换为yolo格式的#############################
#需要设置names，label_path即可完成标签的转换
#转换后的yolo的txt标签在同级xml标签目录下

from xml.dom import minidom
import os
import glob

def convert_coordinates(size, box):
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1]-box[0]
    h = box[3]-box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_xml2yolo( names,label_path ):
    lut={}
    for i,name in enumerate(names):
        lut[name]=i

    for fname in [i for i in os.listdir(label_path) if i.endswith(".xml")]:
        fname=os.path.join(label_path,fname)
        xmldoc = minidom.parse(fname)
        fname_out = (fname[:-4]+'.txt')

        with open(fname_out, "w") as f:
            itemlist = xmldoc.getElementsByTagName('object')
            size = xmldoc.getElementsByTagName('size')[0]
            width = int((size.getElementsByTagName('width')[0]).firstChild.data)
            height = int((size.getElementsByTagName('height')[0]).firstChild.data)

            for item in itemlist:
                # get class label
                classid =  (item.getElementsByTagName('name')[0]).firstChild.data
                if classid in lut:
                    label_str = str(lut[classid])
                else:
                    label_str = "-1"
                    print ("warning: label '%s' not in look-up table" % classid)

                # get bbox coordinates
                xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = convert_coordinates((width,height), b)
                #print(bb)
                f.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')
        print ("wrote %s" % fname_out)

if __name__ == '__main__':
    names=['leaf','branch','bottle','grass','milk_box','plastic_bag',    #如果是yolo格式的需要修改这个列表
         'can','plastic_box','glass_bottle','bad_fruit','ball']
    label_path=r"C:\Users\14833\Desktop\xml"
    convert_xml2yolo( names,label_path)
