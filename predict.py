"""run bash scripts/download_models.sh first to prepare the weights file"""
import os
import shutil
from argparse import Namespace
from applications.cog.first_order import FirstOrderInput, FirstOrderModelDefault, FirstOrderPredictor
from cog import BasePredictor, Input, Path

checkpoints = "checkpoints"
MODEL_SIZE = 256

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("hello world")
        self.firstOrderPredictor = FirstOrderPredictor(output=FirstOrderModelDefault.output,
                                    filename=FirstOrderModelDefault.filename,
                                    weight_path=FirstOrderModelDefault.weight_path,
                                    config=FirstOrderModelDefault.config,
                                    relative=FirstOrderModelDefault.relative,
                                    adapt_scale=FirstOrderModelDefault.adapt_scale,
                                    find_best_frame=FirstOrderModelDefault.find_best_frame,
                                    best_frame=FirstOrderModelDefault.best_frame,
                                    ratio=FirstOrderModelDefault.ratio,
                                    face_detector=FirstOrderModelDefault.face_detector,
                                    multi_person=FirstOrderModelDefault.multi_person,
                                    image_size=FirstOrderModelDefault.image_size,
                                    batch_size=FirstOrderModelDefault.batch_size,
                                    face_enhancement=FirstOrderModelDefault.face_enhancement,
                                    mobile_net=FirstOrderModelDefault.mobile_net,
                                    slice_size=FirstOrderModelDefault.slice_size)

    def predict(
        self,
        source_image: Path = Input(
            description="Upload the source image, it can be picture.png",
        ),
        driving_video: Path = Input(
            description="Upload the driven video, accepts .mp4 file",
        )
    ) -> Path:
        """Run a single prediction on the model"""

        pic_path = str(source_image)
        video_path = str(driving_video)

        print(f"Source image: {pic_path}, Video path: {video_path}")

        # crop image and extract 3dmm from image
        results_dir = "results"
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        
        self.firstOrderPredictor.run(
            source_image=pic_path,
            driving_video=video_path)

        output = "/tmp/out.mp4"
        mp4_path = os.path.join(FirstOrderModelDefault.output, [f for f in os.listdir(FirstOrderModelDefault.output) if "result.mp4" in f][0])
        shutil.copy(mp4_path, output)

        return Path(output)

def load_default():
    return Namespace(
        pose_style=0,
        batch_size=2,
        size=MODEL_SIZE,
        expression_scale=1.0,
        input_yaw=None,
        input_pitch=None,
        input_roll=None,
        enhancer=None,
        background_enhancer=None,
        cpu=False,
        face3dvis=False,
        #still=False,
        #preprocess=crop
        #verbose
        #old_version
        net_recon="resnet50",
        init_path=None,
        use_last_fc=False,
        bfm_folder="./src/config/",
        bfm_model="BFM_model_front.mat",
        focal=1015.0,
        center=112.0,
        camera_d=10.0,
        z_near=5.0,
        z_far=15.0,
    )
