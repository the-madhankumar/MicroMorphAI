from roboflow import Roboflow
rf = Roboflow(api_key="J2Kjc385wifWVqDB1NyR")
project = rf.workspace("new-pvzvq").project("speciessam-bts9f")
version = project.version(2)
dataset = version.download("coco-segmentation")
                