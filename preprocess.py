from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
import os
from skimage.transform import resize
import argparse

def create_dir(path):
  if not os.path.exists(path):
      os.makedirs(path)

def remove_old_images(path):
  filelist = glob.glob(os.path.join(path, "*.png"))
  for f in filelist:
      os.remove(f)

def process_video_frames(video_source_path, fps):
  imagestore = []
  # List of all Videos in the Source Directory.
  videos = os.listdir(video_source_path)
  print("Found", len(videos), "training video")
  # Make a temp dir to store all the frames
  framepath = video_source_path + '/frames'
  create_dir(framepath)
  # Remove old images
  remove_old_images(framepath)
  cnt = 0
  for video in videos:
    create_dir(os.path.join(framepath, str(cnt)))
    os.system('ffmpeg -i {}/{} -vf fps={} {}/frames/{}/%04d.jpg'.format(video_source_path, video, fps, video_source_path, cnt))
    images = os.listdir(os.path.join(framepath, str(cnt)))
    for image in images:
        image_path = os.path.join(framepath, str(cnt), image)
        store_image(image_path, imagestore)
    cnt += 1
  imagestore = preprocess_images(imagestore)
  np.save('npys/training.npy', imagestore)
  # Remove Buffer Directory
  os.system('rm -r {}'.format(framepath))

def store_image(image_path, imagestore):
  img = load_img(image_path)
  img = img_to_array(img)
  # Resize the Image to (227,227,3) for the network to be able to process it.
  img = resize(img, (227, 227, 3))
  # Convert the Image to Grayscale
  gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
  imagestore.append(gray)

def preprocess_images(imagestore):
  imagestore = np.array(imagestore)
  a, b, c = imagestore.shape
  # Reshape to (227, 227, batch_size)
  imagestore.resize(b, c, a)
  # Normalize
  imagestore = (imagestore - imagestore.mean()) / (imagestore.std())
  # Clip negative Values
  imagestore = np.clip(imagestore, 0, 1)
  return imagestore

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Source Video path')
  parser.add_argument('source_vid_path', type=str)
  parser.add_argument('fps', type=int)
  args = parser.parse_args()
  video_source_path = args.source_vid_path
  fps = args.fps
  process_video_frames(video_source_path, fps)