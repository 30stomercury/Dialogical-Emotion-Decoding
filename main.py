from ded import beam_search as bs
from ded.arguments import parse_args
from ded import utils
import os
import sys
import numpy as np
import joblib
import logging
import json


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
  """
  dialogs: dict, dialogs of the dataset.
  emo_dict: dict, emotions of all utterances.
  out_dict: dict, output logits of emotion classifier.
  """
  dialogs = joblib.load('./data/dialog_iemocap.pkl')
  emo_dict = joblib.load('./data/emo_all_iemocap.pkl')
  out_dict = joblib.load('./data/outputs.pkl')

  # set log
  logging.basicConfig(stream=sys.stdout,
                      format='%(asctime)s %(levelname)s:%(message)s',
                      level=logging.INFO,
                      datefmt='%I:%M:%S')

  # Split dialogs by speakers
  spk_dialogs = utils.split_dialog(dialogs)
  logging.info("Average number of speaker's utterances per dialog: %.3f" % np.mean(
                                                    [len(i) for i in spk_dialogs.values()]))


  # arguments
  args = parse_args()

  print('=' * 60 + '\n')
  logging.info('Parameters are:\n%s\n', json.dumps(vars(args), sort_keys=False, indent=4))
  print('=' * 60 + '\n')

  if args.transition_bias > 0:
    # Use given p_0
    p_0 = args.transition_bias

  else:
    # Estimate p_0 of ALL dialogs.
    p_0, total_transit = utils.transition_bias(spk_dialogs, emo_dict)

    print("\n"+"#"*50)
    logging.info('p_0: %.3f , total transition: %d\n' % (p_0, total_transit))
    print("#"*50)

    bias_dict = utils.get_val_bias(spk_dialogs, emo_dict)
    print("#"*50+"\n")

  trace = []
  label = []
  org_pred = []
  DED = bs.BeamSearch(p_0, args.crp_alpha, args.num_state, 
                              args.beam_size, args.test_iteration, emo_dict, out_dict)

  for i, dia in enumerate(dialogs):
    logging.info("Decoding dialog: {}/{}, {}".format(i,len(dialogs),dia))

    # Apply p_0 estimated from other 4 sessions.
    DED.transition_bias = bias_dict[dia[:5]] 
    
    # Beam search decoder
    out = DED.decode(dialogs[dia]) 

    trace += out
    label += [utils.convert_to_index(emo_dict[utt]) for utt in dialogs[dia]]
    org_pred += [np.argmax(out_dict[utt]) for utt in dialogs[dia]]
    if args.verbosity > 0:
      logging.info("Output: {}\n".format(out))

  print("#"*50+"\n")
  # Print the results of emotino classifier module
  uar, acc, conf = utils.evaluate(org_pred, label)
  logging.info('Original performance: uar: %.3f, acc: %.3f' % (uar, acc))

  # Eval ded outputs
  results = vars(args)
  uar, acc, conf = utils.evaluate(trace, label)
  logging.info('DED performance: uar: %.3f, acc: %.3f' % (uar, acc))
  logging.info('Confusion matrix:\n%s' % conf)

  # Save results
  results['uar'] = uar
  results['acc'] = acc
  results['conf'] = str(conf)
  logging.info('Save results:')
  logging.info('\n%s\n', json.dumps(results, sort_keys=False, indent=4))
  json.dump(results, open(args.out_dir+'/%s.json' % args.result_file, "w"))


if __name__ == '__main__':
    main()
