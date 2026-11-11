import argparse
from contactsplit.predictor import ContactSplitPredictor,save_result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--output',default='outputs/prediction')
    p.add_argument('--device',default=None)
    a=p.parse_args()
    predictor=ContactSplitPredictor(a.checkpoint,a.device)
    result=predictor.predict_file(a.image)
    save_result(result,a.output)
    print(f"Found {len(result['instances'])} instances. Saved to {a.output}")
