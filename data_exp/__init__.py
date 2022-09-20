from torch.utils.data import DataLoader


from .dataset import COCO
from .field import ImageDetectionsField,TextField,BoxField


def Build_DataSet(args,text_field):


    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=49)

    box_field = BoxField(dect_path=args.dect_path,sc_image_path=args.annotation_folder)
    # Create the dataset
    coco = COCO(args.features_path,args.annotation_folder,fields={'image':image_field, 'box':box_field, 'text':text_field})

    datasets = coco.get_dataset()
    datasets_evalue = coco.get_evalue_dataset()

    return  datasets,datasets_evalue


    

