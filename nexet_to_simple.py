from __future__ import print_function
import os

def invalid_bbox(x1, y1, x2, y2):
    area = abs(x1 - x2) * abs(y1 - y2)
    return area < 70

def nexet_to_simple(train_boxes_csv, train_folder, simple_filename):
    assert(os.path.exists(train_folder))

    with open(train_boxes_csv, 'r') as read_f:
        with open(simple_filename, 'w') as write_f:
            for line in read_f:
                line_split = line.strip().split(',')
                (filename, x1, y1, x2, y2, class_name, confidence) = line_split
                fullpath = os.path.join(train_folder, filename)
                if os.path.exists(fullpath):
                    if ',' in fullpath:
                        print('warning path contains comma:', fullpath, 'line:', line)
                    else:
                        x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
                        if invalid_bbox(x1, y1, x2, y2):
                            print('warning: rejecting invalid bbox')
                        else:
                            simple_line = '{0},{1},{2},{3},{4},{5}'.format(fullpath, x1, y1, x2, y2, class_name)
                            print(simple_line, file = write_f)
                else:
                    print('warning non-existent file:', fullpath, 'line:', line)

def convert():
    nexet_to_simple('/home/eljefec/data/nexet/train_boxes_60c87b87.csv',
                    '/home/eljefec/data/nexet/train',
                    '/home/eljefec/data/nexet/train_boxes_60c87b87.simple.csv')

if __name__ == '__main__':
    convert()
