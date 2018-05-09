from styx_msgs.msg import TrafficLight
import numpy as np
import cv2

RED_THRESHOLD = 200
GREEN_THRESHOLD = 100
THRESHOLD = 0.5
class TLClassifier(object):
    def __init__(self, is_sim):
        #TODO load classifier

        self.is_sim = is_sim

        if not self.is_sim:
            import tensorflow as tf
            PATH_TO_CKPT = "light_classification/model/frozen_inference_graph.pb"
            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

            self.sess = tf.Session(graph=self.detection_graph)



    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if self.is_sim:
            #TODO implement light color prediction
            b, g, r = cv2.split(image)
            r[r <= RED_THRESHOLD] = 0
            r[r > RED_THRESHOLD] = 255
            g[g > GREEN_THRESHOLD] = 255
            g[g <= GREEN_THRESHOLD] = 0
            g = 255 - g

            mask = np.array(r) * (np.array(g) / 255)
            num_red = np.sum(mask / 255)
            mask = cv2.merge((mask, 0 * r, 0 * r))

            if num_red >= 40:
                return TrafficLight.RED
            else:
                return TrafficLight.UNKNOWN
        else:
            with self.detection_graph.as_default():
                image_np_expanded = np.expand_dims(image, axis=0)

                (boxes, scores, classes, num) = self.sess.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: image_np_expanded})

                classes = np.squeeze(classes).astype(np.int32)
                scores = np.squeeze(scores)

                if scores[0] > THRESHOLD:
                    if classes[0] == 2:
                        return TrafficLight.RED
                    else:
                        return TrafficLight.UNKNOWN
