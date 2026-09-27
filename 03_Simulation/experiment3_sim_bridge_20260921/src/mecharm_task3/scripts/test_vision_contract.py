#!/usr/bin/env python3
"""Unit checks for image detection and grid association; not simulation acceptance."""
import unittest
from pathlib import Path
import cv2
import numpy as np
from vision_msgs.msg import Detection2D,Detection2DArray,ObjectHypothesisWithPose
from common import load_config,project_xy,grid_for_pixel,snapshot_grid
from detector import detect_hsv

CONFIG=load_config(Path(__file__).resolve().parents[1]/'config/sorting.yaml')

class VisionContractTest(unittest.TestCase):
    def test_all_six_grid_centres_and_background(self):
        for key,row in CONFIG['grids'].items():
            self.assertEqual(grid_for_pixel(CONFIG,*project_xy(CONFIG,row['xy'])),key)
        self.assertIsNone(grid_for_pixel(CONFIG,0,0))

    def test_actual_pixels_determine_three_classes(self):
        frame=np.full((720,960,3),80,np.uint8)
        cv2.rectangle(frame,(100,100),(160,160),(0,0,240),-1)
        cv2.rectangle(frame,(250,100),(310,160),(240,0,0),-1)
        cv2.rectangle(frame,(400,100),(460,160),(0,240,0),-1)
        boxes=detect_hsv(frame,CONFIG)
        self.assertEqual({x[0] for x in boxes},{'red_cube','blue_cube','unknown'})
        self.assertTrue(all(x[1]>0.9 for x in boxes))

    def test_empty_scene_and_noise_are_not_objects(self):
        frame=np.full((720,960,3),80,np.uint8)
        frame[20:23,20:23]=(0,0,255)
        self.assertEqual(detect_hsv(frame,CONFIG),[])

    def test_rotated_cube_is_not_rejected_as_low_confidence(self):
        for angle in [0,30,45,65,103]:
            frame=np.full((720,960,3),80,np.uint8)
            polygon=np.round(cv2.boxPoints(((400,300),(60,60),angle))).astype(np.int32)
            cv2.fillConvexPoly(frame,polygon,(0,0,240))
            boxes=detect_hsv(frame,CONFIG)
            self.assertEqual(len(boxes),1)
            self.assertEqual(boxes[0][0],'red_cube')
            self.assertGreater(boxes[0][1],0.90)

    def test_unknown_and_low_confidence_survive_message_contract(self):
        msg=Detection2DArray();d=Detection2D();u,v=project_xy(CONFIG,CONFIG['grids']['2']['xy'])
        d.bbox.center.position.x=u;d.bbox.center.position.y=v;d.bbox.size_x=40.;d.bbox.size_y=40.
        h=ObjectHypothesisWithPose();h.hypothesis.class_id='unknown';h.hypothesis.score=0.2;d.results=[h];msg.detections=[d]
        grid=snapshot_grid(CONFIG,msg)
        self.assertEqual(grid['2'][0]['class_id'],'unknown');self.assertEqual(grid['2'][0]['score'],0.2)
        self.assertEqual(grid['1'],[])

if __name__=='__main__':unittest.main()
