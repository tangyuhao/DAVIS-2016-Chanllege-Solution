from SSD500 import *
from collections import Counter
import operator
import cv2
from collections import OrderedDict
# output clip_info structure:
# [   
#     [
#         {
#             "class": int,
#             "score": value from 0 to 1,
#             "box": [ymin, xmin, ymax, xmax]
#         },

#         {
#             "class": int,
#             "score": value from 0 to 1,
#             "box": [ymin, xmin, ymax, xmax]
#         },

#         {
#             "class": int,
#             "score": value from 0 to 1,
#             "box": [ymin, xmin, ymax, xmax]
#         },
#         ...
#     ],
#     [
#         {
#             "class": int,
#             "score": value from 0 to 1,
#             "box": [ymin, xmin, ymax, xmax]
#         },

#         {
#             "class": int,
#             "score": value from 0 to 1,
#             "box": [ymin, xmin, ymax, xmax]
#         },

#         {
#             "class": int,
#             "score": value from 0 to 1,
#             "box": [ymin, xmin, ymax, xmax]
#         },
#         ...
#     ]
# ]
class_dict = {
    12: "dog",
    15: "person",
    19: "train",
    7: "car",
    9: "chair",
    13: "horse",
    16: "plant",
    2: "bicycle",
    10: "cow",
    3: "bird",
    14: "motorbike",
    4: "boat",
    8: "cat",
    6: "bus",
    1: "plane",
    5: "other",
    11: "other",
    17: "sheep",
    18: "other",
    20: "other"
}
not_possible_classes = [16,9,5,11,18,20]
def get_targets(clip_path):
    '''
    return find_target, max_class(int), class_count, clip_info
    '''
    clip_info, dimension = get_clip_info(clip_path)
    clip_classes = []
    for frame in clip_info:
        frame_info = frame["frame_info"]
        frame_classes = []
        for box_info in frame_info:
            if box_info["class"] not in frame_classes:
                frame_classes.append(box_info["class"])
        clip_classes = clip_classes + frame_classes
    # now get all clip_classes
    # count the time of appearance in whole clip for each class
    class_count = dict(Counter(clip_classes).items())

    print(class_count)
    print("then filter static things")
    for cur_class in class_count.copy():
        if cur_class in not_possible_classes:
            class_count.pop(cur_class)


    if (len(class_count) > 0):
        
        #max_class = max(class_count, key=class_count.get)
        max_classes = OrderedDict(Counter(class_count).most_common(2))
        if (len(max_classes) == 2):
            max_classes_list = list(max_classes.items())
            if (max_classes_list[0][1] * 0.7 > max_classes_list[1][1]):
                # remove the second item
                max_classes.pop(max_classes_list[1][0])
        for max_class in max_classes:
            print("most frequent class: %s" %(class_dict[max_class]))
        return True, max_classes, class_count, clip_info, dimension
    else:
        return False, None, class_count, clip_info, dimension

def get_biggest_class_box(frame_info, target_class):

    target_boxes = [box_info["box"] for box_info in frame_info if box_info["class"] == target_class]
    if not target_boxes:
        return {"have_box": False, "biggest_box": None}
    else:
        max_box = [0,0,0,0]
        for box in target_boxes:
            if (box[2] - box[0]) * (box[3] - box[1]) > (max_box[2] - max_box[0]) * (max_box[3] - max_box[1]):
                max_box = box
        return {"have_box": True, "biggest_box": max_box}

if __name__ == "__main__":
    test_clips_filename = "./test.txt"
    f = open(test_clips_filename, "r")
    test_clips = f.read().splitlines()
    test_prefix = "../BF_Segmentation/DAVIS/images/"
    test_folders = []
    output_prefix = "../BF_Segmentation/DAVIS/SSD_box/"
    for clip in test_clips:
        test_folders.append(os.path.join(test_prefix,clip))
    for clip_path in test_folders:
        have_class ,max_classes, class_count, clip_info, [height, width]= get_targets(clip_path)
        if (have_class):
            for target_class in max_classes:
                print(clip_path+":",class_dict[target_class])
            for frame in clip_info:
                frame_info = frame["frame_info"]
                # it is person or bicycle or a motocycle 
                out_img = np.zeros((height,width,1), np.uint8)
                willoutput = False
                for target_class in max_classes:
                    Bbox_info = get_biggest_class_box(frame_info, target_class)
                    if (Bbox_info["have_box"]):
                        willoutput = True
                        max_box = Bbox_info["biggest_box"] # [ymin, xmin, ymax, xmax]
                        [ymin, xmin, ymax, xmax] = max_box
                        cv2.rectangle(out_img,(int(xmin * width),int(ymin * height)),
                            (int(xmax * width),int(ymax * height)),(255,0,0),-1)
                    # print(outfile_path)
                if (willoutput):
                    outfile_path = os.path.join(output_prefix,frame["frame_name"][:-4]+".png")
                    out_dir = os.path.dirname(outfile_path)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    cv2.imwrite(outfile_path, out_img)
        else:
            print(clip_path+":not found classes")













