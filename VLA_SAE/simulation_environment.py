import pybullet as p
import pybullet_data
import numpy as np
import time


class PyBulletManipulationEnv:
    def __init__(self, gui=True):
        self.gui = gui
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)

        self._load_scene()
        self._setup_camera()

    def _load_scene(self):
        # Plane
        p.loadURDF("plane.urdf")

        # Robot arm (KUKA iiwa – simple + stable)
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
        )

        self.num_joints = p.getNumJoints(self.robot_id)

        # Cube
        self.cube_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=[0.5, 0.0, 0.025],
        )

        # Target region (visual box)
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.05, 0.05, 0.001],
            rgbaColor=[0, 1, 0, 0.6],
        )
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.05, 0.05, 0.001],
        )

        self.target_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0.6, 0.2, 0.001],
        )

    def _setup_camera(self):
        self.img_width = 128
        self.img_height = 128
        self.fov = 60
        self.near = 0.01
        self.far = 2.0

    def get_wrist_camera_image(self):
        # Use last link as wrist
        link_state = p.getLinkState(self.robot_id, self.num_joints - 1)
        pos, orn = link_state[0], link_state[1]

        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        camera_direction = rot_matrix @ np.array([0, 0, 1])
        up_vector = rot_matrix @ np.array([0, 1, 0])

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=pos,
            cameraTargetPosition=pos + 0.1 * camera_direction,
            cameraUpVector=up_vector,
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.img_width / self.img_height,
            nearVal=self.near,
            farVal=self.far,
        )

        img = p.getCameraImage(
            self.img_width,
            self.img_height,
            view_matrix,
            projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb = np.reshape(img[2], (self.img_height, self.img_width, 4))[:, :, :3]
        return rgb

    def step(self, joint_positions):
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=200,
            )
        p.stepSimulation()
        if self.gui:
            time.sleep(1 / 240)

    def cube_in_target(self):
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        target_pos, _ = p.getBasePositionAndOrientation(self.target_id)
        return np.linalg.norm(np.array(cube_pos)[:2] - np.array(target_pos)[:2]) < 0.05

    def close(self):
        p.disconnect()
