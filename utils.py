class ClassInstancesDataset():
    def __init__(self):
        self.class_wise_instance_info = [[] for i in range(81)]
        self.instance_info = []

    def add_instance(self, image_path, mask_, class_id, instance_id):
        instance_file = instance_dir+str(instance_id)+".pickle"
        mask_obj = np.asfortranarray(mask_.astype("uint8"))
        mask_obj = maskUtils.encode(mask_obj)
        self.class_wise_instance_info[class_id].append(
            {"image_path": image_path, "instance_path": instance_file, "mask_obj": mask_obj})
        self.instance_info.append({"image_path": image_path, "instance_path": instance_file,
                                   "class_id": class_id, "mask_obj": mask_obj})
