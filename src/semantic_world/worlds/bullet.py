import logging
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import pycram_bullet as p
import yaml
from typing_extensions import Optional

from ..config.world_conf import WorldConfig
from ..enums import WorldMode
from ..pose import Pose
from ..world import World


@dataclass
class Bullet(World):
    """
    Implementation of World using pybullet physics engine.
    """

    mode: WorldMode = WorldMode.DIRECT
    """
    Can either be "GUI" for rendered window or "DIRECT" for non-rendered. The default parameter is "DIRECT"
    """

    _id: Optional[int] = None
    """
    The ID of the world. This is used to identify the world in pybullet.
    """

    _gui_thread: Optional[threading.Thread] = None
    """
    The thread that runs the GUI. This is used to control the camera.
    """

    def start(self):
        """
        Start the world.
        """
        # This disables file caching from PyBullet, since this would also cache
        # files that can not be loaded

        if self.mode == WorldMode.GUI:
            try:
                self._gui_thread: Gui = Gui()
                self._gui_thread.start()
                time.sleep(0.1)
                self._id = self._gui_thread._id
            except IOError as e:
                logging.warning(str(e))
                self.mode = WorldMode.DIRECT
                self._id = p.connect(p.DIRECT)
        else:
            self._id = p.connect(p.DIRECT)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -9.81, physicsClientId=self._id)

    def spawn_bodies(self):
        """
        Spawn bodies in the world.
        """
        pass

    def load_urdf(self, file_path: str, pose: Pose = Pose()) -> int:
        """
        Load a URDF file into the world.

        :param file_path: The path to the URDF file.
        :param pose: The pose of the object in the world.
        """
        if file_path is None:
            raise ValueError("Path to the object file is required.")
        return p.loadURDF(file_path,
                          basePosition=pose.position.to_list(),
                          baseOrientation=pose.orientation.to_list(), physicsClientId=self._id)


class Gui(threading.Thread):
    """
    For internal use only. Creates a new thread for the physics simulation that is active until closed by
    :func:`~World.exit`
    Also contains the code for controlling the camera.
    """

    def __init__(self):
        threading.Thread.__init__(self)
        self._id: Optional[int] = None
        self.camera_button_id = -1

        # Checks if there is a display connected to the system. If not, the simulation will be run in direct mode.
        if "DISPLAY" not in os.environ:
            raise IOError("No display detected. Running the simulation in direct mode.")

    def run(self):
        """
        Initializes the new simulation and checks in an endless loop
        if it is still active. If it is the thread will be suspended for 1/80 seconds, if it is not the method and
        thus the thread terminates. The loop also checks for mouse and keyboard inputs to control the camera.
        """
        self._id = p.connect(p.GUI)
        self.camera_button_id = p.addUserDebugParameter("Save as Default Camera", 1, 0, 1, physicsClientId=self._id)

        # DISCLAIMER
        # This camera control only works if the WorldMode.GUI BulletWorld is the first one to be created. This is
        # due to a bug in the function pybullet.getDebugVisualizerCamera() which only returns the information of
        # the first created simulation.

        # Disable the side windows of the GUI
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._id)
        # Change the init camera pose
        default_camera_config = WorldConfig.default_camera_config
        p.resetDebugVisualizerCamera(cameraDistance=default_camera_config["dist"],
                                     cameraYaw=default_camera_config["yaw"],
                                     cameraPitch=default_camera_config["pitch"],
                                     cameraTargetPosition=default_camera_config["target_position"],
                                     physicsClientId=self._id)

        # Get the initial camera target location
        camera_target_position = p.getDebugVisualizerCamera(physicsClientId=self._id)[11]

        sphere_visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1],
                                               physicsClientId=self._id)

        # Create a sphere with a radius of 0.05 and a mass of 0
        sphere_uid = p.createMultiBody(baseMass=0.0,
                                       baseInertialFramePosition=[0, 0, 0],
                                       baseVisualShapeIndex=sphere_visual_id,
                                       basePosition=camera_target_position,
                                       physicsClientId=self._id)

        # Define the maxSpeed, used in calculations
        max_speed = 16

        # Set initial Camera Rotation
        camera_yaw = default_camera_config["yaw"]
        camera_pitch = default_camera_config["pitch"]

        # Keep track of the mouse state
        mouse_state = [0, 0, 0]
        old_mouse_x, old_mouse_y = 0, 0

        # Determines if the sphere at cameraTargetPosition is visible
        visible = 1

        # Initial value for the camera button
        last_button_value = 1

        # Loop to update the camera position based on keyboard events
        while p.isConnected(self._id):
            # Check if Camera button was pressed
            camera_button_value = p.readUserDebugParameter(self.camera_button_id)
            if camera_button_value != last_button_value:
                last_button_value = camera_button_value

                current_camera_config = p.getDebugVisualizerCamera()[8:]
                v = dict(zip(["yaw", "pitch", "dist", "target_position"], current_camera_config))
                v["target_position"] = list(v["target_position"])
                yaml_path = os.path.join(os.path.dirname(__file__), "..", "config", 'camera.yaml')
                with open(yaml_path, "w") as f:
                    yaml.dump(v, f)

            # Monitor user input
            keys = p.getKeyboardEvents(self._id)
            mouse = p.getMouseEvents(self._id)

            # Get infos about the camera
            width, height, dist = (p.getDebugVisualizerCamera()[0],
                                   p.getDebugVisualizerCamera()[1],
                                   p.getDebugVisualizerCamera()[10])
            # print("width: ", width, "height: ", height, "dist: ", dist)
            camera_target_position = p.getDebugVisualizerCamera(self._id)[11]

            # Get vectors used for movement on x,y,z Vector
            x_vec = [p.getDebugVisualizerCamera(self._id)[2][i] for i in [0, 4, 8]]
            y_vec = [p.getDebugVisualizerCamera(self._id)[2][i] for i in [2, 6, 10]]
            z_vec = (0, 0, 1)  # [p.getDebugVisualizerCamera()[2][i] for i in [1, 5, 9]]

            # Check the mouse state
            if mouse:
                for m in mouse:

                    mouse_x = m[2]
                    mouse_y = m[1]

                    # update mouseState
                    # Left Mouse button
                    if m[0] == 2 and m[3] == 0:
                        mouse_state[0] = m[4]
                    # Middle mouse button (scroll wheel)
                    if m[0] == 2 and m[3] == 1:
                        mouse_state[1] = m[4]
                    # right mouse button
                    if m[0] == 2 and m[3] == 2:
                        mouse_state[2] = m[4]

                    # change visibility by clicking the mousewheel
                    # if m[4] == 6 and m[3] == 1 and visible == 1:
                    #     visible = 0
                    # elif m[4] == 6 and visible == 0:
                    #     visible = 1

                    # camera movement when the left mouse button is pressed
                    if mouse_state[0] == 3:
                        speed_x = abs(old_mouse_x - mouse_x) if (abs(old_mouse_x - mouse_x)) < max_speed \
                            else max_speed
                        speed_y = abs(old_mouse_y - mouse_y) if (abs(old_mouse_y - mouse_y)) < max_speed \
                            else max_speed

                        # max angle of 89.5 and -89.5 to make sure the camera does not flip (is annoying)
                        if mouse_x < old_mouse_x:
                            if (camera_pitch + speed_x) < 89.5:
                                camera_pitch += (speed_x / 4) + 1
                        elif mouse_x > old_mouse_x:
                            if (camera_pitch - speed_x) > -89.5:
                                camera_pitch -= (speed_x / 4) + 1

                        if mouse_y < old_mouse_y:
                            camera_yaw += (speed_y / 4) + 1
                        elif mouse_y > old_mouse_y:
                            camera_yaw -= (speed_y / 4) + 1

                    # Camera movement when the middle mouse button is pressed
                    if mouse_state[1] == 3:
                        speed_x = abs(old_mouse_x - mouse_x)
                        factor = 0.05

                        if mouse_x < old_mouse_x:
                            dist = dist - speed_x * factor
                        elif mouse_x > old_mouse_x:
                            dist = dist + speed_x * factor
                        dist = max(dist, 0.1)

                    # camera movement when the right mouse button is pressed
                    if mouse_state[2] == 3:
                        speed_x = abs(old_mouse_x - mouse_x) if (abs(old_mouse_x - mouse_x)) < 5 else 5
                        speed_y = abs(old_mouse_y - mouse_y) if (abs(old_mouse_y - mouse_y)) < 5 else 5
                        factor = 0.05

                        if mouse_x < old_mouse_x:
                            camera_target_position = np.subtract(camera_target_position,
                                                                 np.multiply(np.multiply(z_vec, factor), speed_x))
                        elif mouse_x > old_mouse_x:
                            camera_target_position = np.add(camera_target_position,
                                                            np.multiply(np.multiply(z_vec, factor), speed_x))

                        if mouse_y < old_mouse_y:
                            camera_target_position = np.add(camera_target_position,
                                                            np.multiply(np.multiply(x_vec, factor), speed_y))
                        elif mouse_y > old_mouse_y:
                            camera_target_position = np.subtract(camera_target_position,
                                                                 np.multiply(np.multiply(x_vec, factor), speed_y))
                    # update oldMouse values
                    old_mouse_y, old_mouse_x = mouse_y, mouse_x

            # check the keyboard state
            if keys:
                # if shift is pressed, double the speed
                if p.B3G_SHIFT in keys:
                    speed_mult = 5
                else:
                    speed_mult = 2.5

                # if control is pressed, the movements caused by the arrowkeys, the '+' as well as the '-' key
                # change
                if p.B3G_CONTROL in keys:

                    # the up and down arrowkeys cause the targetPos to move along the z axis of the map
                    if p.B3G_DOWN_ARROW in keys:
                        camera_target_position = np.subtract(camera_target_position,
                                                             np.multiply(np.multiply(z_vec, 0.03), speed_mult))
                    elif p.B3G_UP_ARROW in keys:
                        camera_target_position = np.add(camera_target_position,
                                                        np.multiply(np.multiply(z_vec, 0.03), speed_mult))

                    # left and right arrowkeys cause the targetPos to move horizontally relative to the camera
                    if p.B3G_LEFT_ARROW in keys:
                        camera_target_position = np.subtract(camera_target_position,
                                                             np.multiply(np.multiply(x_vec, 0.03), speed_mult))
                    elif p.B3G_RIGHT_ARROW in keys:
                        camera_target_position = np.add(camera_target_position,
                                                        np.multiply(np.multiply(x_vec, 0.03), speed_mult))

                    # the '+' and '-' keys cause the targetpos to move forwards and backwards relative to the camera
                    # while the camera stays at a constant distance. SHIFT + '=' is for US layout
                    if ord("+") in keys or p.B3G_SHIFT in keys and ord("=") in keys:
                        camera_target_position = np.subtract(camera_target_position,
                                                             np.multiply(np.multiply(y_vec, 0.03), speed_mult))
                    elif ord("-") in keys:
                        camera_target_position = np.add(camera_target_position,
                                                        np.multiply(np.multiply(y_vec, 0.03), speed_mult))

                # standard bindings for thearrowkeys, the '+' as well as the '-' key
                else:

                    # left and right arrowkeys cause the camera to rotate around the yaw axis
                    if p.B3G_RIGHT_ARROW in keys:
                        camera_yaw += (360 / width) * speed_mult
                    elif p.B3G_LEFT_ARROW in keys:
                        camera_yaw -= (360 / width) * speed_mult

                    # the up and down arrowkeys cause the camera to rotate around the pitch axis
                    if p.B3G_DOWN_ARROW in keys:
                        if (camera_pitch + (360 / height) * speed_mult) < 89.5:
                            camera_pitch += (360 / height) * speed_mult
                    elif p.B3G_UP_ARROW in keys:
                        if (camera_pitch - (360 / height) * speed_mult) > -89.5:
                            camera_pitch -= (360 / height) * speed_mult

                    # the '+' and '-' keys cause the camera to zoom towards and away from the targetPos without
                    # moving it. SHIFT + '=' is for US layout since the events can't handle shift plus something
                    if ord("+") in keys or p.B3G_SHIFT in keys and ord("=") in keys:
                        if (dist - (dist * 0.02) * speed_mult) > 0.1:
                            dist -= dist * 0.02 * speed_mult
                    elif ord("-") in keys:
                        dist += dist * 0.02 * speed_mult
            # print("dist: ", dist)
            # print("camera_yaw: ", camera_yaw)
            # print("camera_pitch: ", camera_pitch)
            # print("camera_target_position: ", camera_target_position)

            p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=camera_yaw, cameraPitch=camera_pitch,
                                         cameraTargetPosition=camera_target_position, physicsClientId=self._id)
            if visible == 0:
                camera_target_position = (0.0, -50, 50)
            p.resetBasePositionAndOrientation(sphere_uid, camera_target_position, [0, 0, 0, 1],
                                              physicsClientId=self._id)
            time.sleep(1. / 80.)
