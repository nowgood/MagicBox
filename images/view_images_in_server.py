import os
import sys
import argparse


def showImageInHTML(imageTypes, dirname):
    files = getAllFiles(dirname)
    images = [f for f in files if f[f.rfind('.') + 1:] in imageTypes]
    images = [item for item in images if os.path.getsize(item) > 5 * 1024]
    images = [
        os.path.basename(dirname) + item[item.rfind('/'):] for item in images
    ]
    newfile = dirname + '.html'
    with open(newfile, 'w') as f:
        f.write('<div>')
        for image in images:
            f.write("<img src='%s'>\n" % image)
        f.write('</div>')
    print('success, images are wrapped up in %s' % newfile)


def getAllFiles(directory):
    files = []
    for dirpath, _, filenames in os.walk(directory):
        if filenames != []:
            for file in filenames:
                files.append(dirpath + '/' + file)
    files.sort(key=len)
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dirname',
        type=str,
        help="The dirname with images, don't endwith / .")
    args = parser.parse_args()

    showImageInHTML(('jpg', 'png', 'gif'), args.dirname)
