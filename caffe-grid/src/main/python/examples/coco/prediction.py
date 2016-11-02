from examples.coco.retrieval_experiment import *

def predict_caption(list_of_images, model, imagenet, lstmnet, vocab):
  out_iterator = []
  ce = CaptionExperiment(str(model),str(imagenet),str(lstmnet),str(vocab))
  for image in list_of_images:
    out_iterator.append(ce.getCaption(image))
  return iter(out_iterator)
