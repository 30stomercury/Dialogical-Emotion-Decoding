import numpy as np

def split_dialog(dialog):
  """Split utterances in a dialog into a set of speaker's utternaces in that dialog.
     See eq (5) in the paper.
  Arg:
    dialog: dict, for example, utterances in dialog with index = 01: 
            {dialog_01: [utt_spk01_1, utt_spk02_1, utt_spk01_2, ...]}.
  Return:
    spk_dialog: dict, a collection of speakers' utterances in dialogs. for example:
            {dialog_01_spk01: [utt_spk01_1, utt_spk01_2, ...],
             dialog_01_spk02: [utt_spk02_1, utt_spk02_2, ...]}
  """

  spk_dialog = {}
  for dialog_id in dialog.keys():
    spk_dialog[dialog_id+'_M'] = []
    spk_dialog[dialog_id+'_F'] = []
    for utt_id in dialog[dialog_id]:
      if utt_id[-4] == 'M':
        spk_dialog[dialog_id+'_M'].append(utt_id)
      elif utt_id[-4] == 'F':
        spk_dialog[dialog_id+'_F'].append(utt_id)

  print("Average of the number of speaker's utterances:", np.mean([len(i) for i in spk_dialog.values()]))
  return spk_dialog

def transition_bias(spk_dialog, emo):
  """Estimate the transition bias of emotion. See eq (5) in the paper.
  Args:
    spk_dialog: dict, a collection of speakers' utterances in dialogs. for example:
    emo: dict, map from utt_id to emotion state.

  Returns: 
    bias: p_0 in eq (4).
  """
  transit_num = 0
  total_transit = 0
  count = 0
  num = 0
  for dialog_id in spk_dialog.values():
      for entry in range(len(dialog_id) - 1):
        transit_num += (emo[dialog_id[entry]] != emo[dialog_id[entry + 1]])
      total_transit += (len(dialog_id) - 1)

      count += len(np.unique([emo[i] for i in dialog_id if emo[i]]))
      if len(np.unique([emo[i] for i in dialog_id if emo[i]])) > 0:
        num += 1

  bias = (transit_num + 1) / total_transit

  print('Ave emotion states in a dialog:', count/num)

  return bias, total_transit

def transition_bias_v2(dialog, emo, val):
  # compute bias
  transit_num = 0
  total_transit = 0
  
  for dialog_id in dialog.values():
      if val in dialog_id[0]:
        continue
      for entry in range(len(dialog_id) - 1):
        #print(dialog_id[entry])
        transit_num += (emo[dialog_id[entry]] != emo[dialog_id[entry + 1]])
      total_transit += len(dialog_id) - 1
  bias = (transit_num + 1) / total_transit

  return bias, total_transit

def find_last_idx(trace_speakers, speaker):
  for i in range(len(trace_speakers)):
    if trace_speakers[len(trace_speakers) - (i+1)] == speaker:
        return len(trace_speakers) - (i+1)

if __name__ == '__main__':
    dialog = {'Ses05M_script03_2_M': ['Ses05M_script03_2_M042', 'Ses05M_script03_2_M043', 
                'Ses05M_script03_2_M044', 'Ses05M_script03_2_M045']}
    emo = {'Ses05M_script03_2_M042': 'ang', 'Ses05M_script03_2_M043': 'ang', 
                'Ses05M_script03_2_M044': 'ang', 'Ses05M_script03_2_M045': 'ang'}

    spk_dialog = split_dialog(dialog)
    bias, total_transit = transition_bias(spk_dialog, emo)
    crp_alpha = 1
    print('Transition bias: {} , Total transition: {}'.format(bias, total_transit))
