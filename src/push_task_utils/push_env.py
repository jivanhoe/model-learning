import cv2
import numpy as np
import torch
from airobot import Robot
from airobot.utils.common import euler2quat, quat2euler


class PushEnv(object):
    """
    Poking environment: loads initial object on a table
    """

    def __init__(self, ifRender=False):
        # np.set_printoptions(precision=3, suppress=True)
        # table scaling and table center location
        self.table_scaling = 0.6  # tabletop x ~ (0.3, 0.9); y ~ (-0.45, 0.45)
        self.table_x = 0.6
        self.table_y = 0
        self.table_z = 0.6
        self.table_surface_height = 0.975  # get from running robot.cam.get_pix.3dpt
        self.table_ori = euler2quat([0, 0, np.pi / 2])
        # task space x ~ (0.4, 0.8); y ~ (-0.3, 0.3)
        self.max_arm_reach = 0.91
        self.workspace_max_x = 0.75  # 0.8 discouraged, as box can move past max arm reach
        self.workspace_min_x = 0.4
        self.workspace_max_y = 0.3
        self.workspace_min_y = -0.3
        # robot end-effector
        self.ee_min_height = 0.99
        self.ee_rest_height = 1.1  # stick scale="0.0001 0.0001 0.0007"
        self.ee_home = [self.table_x, self.table_y, self.ee_rest_height]
        # initial object location
        self.box_z = 1 - 0.005
        self.box_pos = [self.table_x, self.table_y, self.box_z]
        self.box_size = 0.02  # distance between center frame and size, box size 0.04
        # push config: push_len by default [0.06-0.1]
        self.push_len_min = 0.06  # 0.06 ensures no contact with box empiracally
        self.push_len_range = 0.04
        # image processing config
        self.row_min = 40
        self.row_max = 360
        self.col_min = 0
        self.col_max = 640
        self.output_row = 100
        self.output_col = 200
        self.row_scale = (self.row_max - self.row_min) / float(self.output_row)
        self.col_scale = (self.col_max - self.col_min) / float(self.output_col)
        assert self.col_scale == self.row_scale
        # load robot
        self.robot = Robot('ur5e_stick', pb=True, pb_cfg={'gui': ifRender})
        self.robot.arm.go_home()
        self.ee_origin = self.robot.arm.get_ee_pose()
        self.go_home()
        self._home_jpos = self.robot.arm.get_jpos()
        # load table
        self.table_id = self.load_table()
        # load box
        self.box_id = self.load_box()
        # initialize camera matrices
        self.robot.cam.setup_camera(focus_pt=[0.7, 0, 1.],
                                    dist=0.5, yaw=90, pitch=-60, roll=0)
        self.ext_mat = self.robot.cam.get_cam_ext()
        self.int_mat = self.robot.cam.get_cam_int()

    def move_ee_xyz(self, delta_xyz):
        return self.robot.arm.move_ee_xyz(delta_xyz, eef_step=0.015)

    def set_ee_pose(self, pos, ori=None, ignore_physics=False):
        jpos = self.robot.arm.compute_ik(pos, ori)
        return self.robot.arm.set_jpos(jpos, wait=True, ignore_physics=ignore_physics)

    def go_home(self):
        self.set_ee_pose(self.ee_home, self.ee_origin[1])

    def tele_home(self):
        # directly use joint values as solving ik may return different values
        return self.robot.arm.set_jpos(position=self._home_jpos, ignore_physics=True)

    def load_table(self):
        return self.robot.pb_client.load_urdf('table/table.urdf',
                                              [self.table_x, self.table_y, self.table_z],
                                              self.table_ori,
                                              scaling=self.table_scaling)

    def load_box(self, pos=None, quat=None, rgba=[1, 0, 0, 1]):
        if pos is None:
            pos = self.box_pos
        return self.robot.pb_client.load_geom('box', size=self.box_size,
                                              mass=1,
                                              base_pos=pos,
                                              base_ori=quat,
                                              rgba=rgba)

    def reset_box(self, box_id=None, pos=None, quat=None):
        if box_id is None:
            box_id = self.box_id
        if pos is None:
            pos = self.box_pos
        return self.robot.pb_client.reset_body(box_id, pos, quat)

    def remove_box(self, box_id):
        self.robot.pb_client.remove_body(box_id)

    def get_ee_pose(self):
        return self.robot.arm.get_ee_pose()

    def get_box_pose(self, box_id=None):
        if box_id is None:
            box_id = self.box_id
        pos, quat, lin_vel, _ = self.robot.pb_client.get_body_state(box_id)
        rpy = quat2euler(quat=quat)
        return pos, quat, rpy, lin_vel

    def get_img(self, resize=True):
        rgb, _ = self.robot.cam.get_images(get_rgb=True)
        if resize:
            rgb = self.resize_rgb(rgb)
        return rgb / 255.0

    def resize_rgb(self, rgb):
        img = rgb[self.row_min:self.row_max, self.col_min:self.col_max]  # type int64
        resized_img = cv2.resize(img.astype('float32'),
                                 dsize=(self.output_col, self.output_row),
                                 interpolation=cv2.INTER_CUBIC)
        return resized_img

    def calculate_push_locations(self, state, push_ang, push_len):
        obj_x, obj_y = state
        start_x = obj_x - self.push_len_min * np.cos(push_ang)
        start_y = obj_y - self.push_len_min * np.sin(push_ang)
        end_x = obj_x + push_len * np.cos(push_ang)
        end_y = obj_y + push_len * np.sin(push_ang)
        return start_x, start_y, end_x, end_y

    def push_is_feasible(self, start_x, start_y, end_x, end_y):
        start_radius = np.sqrt(start_x ** 2 + start_y ** 2)
        end_radius = np.sqrt(end_x ** 2 + end_y ** 2)
        return (
            (start_radius < self.max_arm_reach)
            and (end_radius + self.push_len_min < self.max_arm_reach)
            and (self.workspace_min_x < end_x < self.workspace_max_x)
            and (self.workspace_min_y < end_y < self.workspace_max_y)
        )

    def sample_push(self, obj_x, obj_y, push_ang=None, push_len=None):
        while True:
            # choose push angle along the z axis
            if push_ang is None:
                push_ang = np.random.random() * np.pi * 2 - np.pi
            # choose push length
            if push_len is None:
                push_len = np.random.random() * self.push_len_range + self.push_len_min
            # calc starting push location and ending push loaction
            start_x, start_y, end_x, end_y = self.calculate_push_locations((obj_x, obj_y), push_ang, push_len)

            # find valid push that does not lock the arm
            if self.push_is_feasible(start_x, start_y, end_x, end_y):
                break
        return start_x, start_y, end_x, end_y

    def execute_push(self, start_x, start_y, end_x, end_y):
        # move to starting push location
        init_obj = self.get_box_pose()[0][:2]
        self.move_ee_xyz([start_x - self.ee_home[0], start_y - self.ee_home[1], 0])
        self.move_ee_xyz([0, 0, self.ee_min_height - self.ee_rest_height])

        self.move_ee_xyz([end_x - start_x, end_y - start_y, 0])  # push
        # important that we use move_ee_xyz, as set_ee_pose can throw obj in motion
        self.move_ee_xyz([0, 0, self.ee_rest_height - self.ee_min_height])

        # move arm away from camera view
        self.go_home()  # important to have one set_ee_pose every loop to reset accu errors
        final_obj = self.get_box_pose()[0][:2]
        return init_obj, final_obj

    def sample_multi_step_push(self, n_steps=2, seed=0):
        self.go_home()
        self.reset_box()
        np.random.seed(seed)
        init_obj = self.get_box_pose()[0][:2]
        push_ang = np.pi / 3 + np.random.random() * np.pi / 3

        if np.random.random() < 0.5:
            push_ang -= np.pi

        goal_obj = None
        pushes = []
        for _ in range(n_steps):
            obj_x, obj_y = self.get_box_pose()[0][:2]
            start_x, start_y, end_x, end_y = self.sample_push(obj_x=obj_x, obj_y=obj_y, push_len=0.1, push_ang=push_ang)
            pushes.append(np.array([start_x, start_y, end_x, end_y]))
            _, goal_obj = self.execute_push(start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y)

        self.reset_box()
        return np.array(init_obj), np.array(goal_obj), pushes
