import os
import gym
import PIL
from PIL import Image
#import cv2

from pyvirtualdisplay import Display
Display().start()

class RecordingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(RecordingWrapper, self).__init__(env)
        self.output_directory = None
    def observation(self, obs):
        if self.output_directory is not None:
            img = self.env.render('rgb_array', close=False)
            #img = self.env.render('rgb_array')
            Image.fromarray(img).save(os.path.join(self.output_directory,'frame-%d.png'%(self.frame_count)))
            self.frame_count += 1
        return obs
    def record_to(self, directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)
        else:
            pass # TODO: Check if directory is empty. If not, clear it.
        self.output_directory = directory
        self.frame_count = 0
    def stop_recording(self):
        #self.save_video()
        self.output_directory = None
    #def save_video(self, video_file_name='video.avi'):
    #    images = [img for img in os.listdir(self.output_directory) if img.endswith(".png")]
    #    frame = cv2.imread(os.path.join(self.output_directory, images[0]))
    #    height, width, layers = frame.shape

    #    video = cv2.VideoWriter(video_file_name, 0, 1, (width,height))

    #    for image in images:
    #        video.write(cv2.imread(os.path.join(image_folder, image)))

    #        cv2.destroyAllWindows()
    #        video.release()
