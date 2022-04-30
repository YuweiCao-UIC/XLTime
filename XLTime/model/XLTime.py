from transformers import AutoModel, AutoTokenizer, XLMRobertaModel, XLMRobertaTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

class XLTime_mBERT(nn.Module):
    """
    Apply the proposed XLTime framework on the mBERT backbone.
    Args:
        n_labels: the number of classes in the sequence labeling task
        label_ignore_idx: the index of the class to be ignored in the sequence labeling task
    """
    def __init__(self, n_labels, hidden_size, dropout_p, label_ignore_idx=0,
                head_init_range=0.04, device='cuda'):
        super().__init__()

        # for the sequence labeling task (task = 0)
        self.n_labels = n_labels
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.classification_head_1 = nn.Linear(hidden_size, n_labels)
        self.label_ignore_idx = label_ignore_idx

        # for the binary classification task (task = 1)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.classification_head_2 = nn.Linear(hidden_size, 2)

        # load the pretrained model and its tokenizer
        self.model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        self.dropout = nn.Dropout(dropout_p)
        self.device=device

        # initializing classification heads
        self.classification_head_1.weight.data.normal_(mean=0.0, std=head_init_range)
        self.classification_head_2.weight.data.normal_(mean=0.0, std=head_init_range)

    def forward(self, inputs_ids, labels, labels_mask, task = 0):
        """
        Computes a forward pass through the model.
        Args:
            inputs_ids: tensor of size (bsz, max_seq_len).
            labels: tensor of size (bsz, max_seq_len).
            labels_mask: indicate where loss gradients should be propagated and where labels should be ignored.
            task: 0 indicates the sequence labeling task, while 1 indicates the binary classification task. 
        """
        if task == 0:
            #print(inputs_ids)
            #print(inputs_ids.size())

            # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) 
            # – Sequence of hidden-states at the output of the last layer of the model.
            transformer_out = self.model(inputs_ids)[0]
            #print(transformer_out)
            out_1 = F.relu(self.linear_1(transformer_out))
            out_1 = self.dropout(out_1)
            logits = self.classification_head_1(out_1)
        
            if labels is not None: # for training
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_ignore_idx) # nn.CrossEntropyLoss(input, target). ignore_index: target value to be ignored. This keeps the active parts of the loss
                # Only keep active parts of the loss
                if labels_mask is not None:
                    active_loss = labels_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.n_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                    #print("Preds = ", active_logits.argmax(dim=-1))
                    #print("Labels = ", active_labels)
                else:
                    loss = loss_fct(
                        logits.view(-1, self.n_labels), labels.view(-1))
                return loss, logits, active_logits.argmax(dim=-1), active_labels # loss, logits, predictions, labels
            else: # for evaluation
                return logits

        elif task == 1:
            # pooler_output (torch.FloatTensor of shape (batch_size, hidden_size))
            transformer_out = self.model(inputs_ids)[1] 
            out_2 = F.relu(self.linear_2(transformer_out))
            out_2 = self.dropout(out_2)
            logits = self.classification_head_2(out_2)

            if labels is not None: # for training
                loss_fct = nn.CrossEntropyLoss()
                logits = logits.view(-1, 2)
                loss = loss_fct(logits, labels.view(-1))
                return loss, logits, logits.argmax(dim=-1), labels.view(-1) # loss, logits, predictions, labels
            else: # for evaluation
                return logits

        else:
            print('Invalid task. task has to be 0 or 1')

    def encode_word(self, s):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.tokenizer.encode(s)
        # remove [CLS] and [SEP] ids
        return tensor_ids[1:-1]
    
    ## BERT: bos_token='[CLS]':101, eos_token='[SEP]':102, pad_token='[PAD]':0 
    def get_special_token_ids(self):
        return {"bos_token_id":101, "eos_token_id":102, "pad_token_id":0}


class XLTime_XLMR(nn.Module):
    """
    Apply the proposed XLTime framework on the XLMR backbone.
    """
    def __init__(self, model_type, n_labels, hidden_size, dropout_p, label_ignore_idx=0,
                head_init_range=0.04, device='cuda'):
        super().__init__()

        # for the sequence labeling task (task = 0)
        self.n_labels = n_labels
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.classification_head_1 = nn.Linear(hidden_size, n_labels)
        self.label_ignore_idx = label_ignore_idx

        # for the binary classification task (task = 1)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.classification_head_2 = nn.Linear(hidden_size, 2)

        # load the pretrained model
        self.model = XLMRobertaModel.from_pretrained(model_type)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_type)

        self.dropout = nn.Dropout(dropout_p)
        self.device=device

        # initializing classification head
        self.classification_head_1.weight.data.normal_(mean=0.0, std=head_init_range)
        self.classification_head_2.weight.data.normal_(mean=0.0, std=head_init_range)

    def forward(self, inputs_ids, labels, labels_mask, task = 0):
        """
        Computes a forward pass through the model.
        Args:
            inputs_ids: tensor of size (bsz, max_seq_len).
            labels: tensor of size (bsz, max_seq_len).
            labels_mask: indicate where loss gradients should be propagated and where labels should be ignored.
            task: 0 indicates the sequence labeling task, while 1 indicates the binary classification task. 
        """
        if task == 0:
            # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) 
            # – Sequence of hidden-states at the output of the last layer of the model.
            transformer_out = self.model(inputs_ids)[0] 
            #print(transformer_out)
            out_1 = F.relu(self.linear_1(transformer_out))
            out_1 = self.dropout(out_1)
            logits = self.classification_head_1(out_1)
        
            if labels is not None: # for training
                # nn.CrossEntropyLoss(input, target). ignore_index: target value to be ignored. 
                # This keeps the active parts of the loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_ignore_idx) 
                # Only keep active parts of the loss
                if labels_mask is not None:
                    active_loss = labels_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.n_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                    #print("Preds = ", active_logits.argmax(dim=-1))
                    #print("Labels = ", active_labels)
                else:
                    loss = loss_fct(
                        logits.view(-1, self.n_labels), labels.view(-1))
                return loss, logits, active_logits.argmax(dim=-1), active_labels # loss, logits, predictions, labels
            else: # for evaluation
                return logits

        elif task == 1:
            # pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) 
            transformer_out = self.model(inputs_ids)[1] 
            out_2 = F.relu(self.linear_2(transformer_out))
            out_2 = self.dropout(out_2)
            logits = self.classification_head_2(out_2)

            if labels is not None: # for training
                loss_fct = nn.CrossEntropyLoss()
                logits = logits.view(-1, 2)
                loss = loss_fct(logits, labels.view(-1))
                return loss, logits, logits.argmax(dim=-1), labels.view(-1) # loss, logits, predictions, labels
            else: # for evaluation
                return logits

        else:
            print('Invalid task. task has to be 0 or 1')

    def encode_word(self, s):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.tokenizer.encode(s)
        # remove <s> and </s> ids
        return tensor_ids[1:-1]
    
    # XLM-R:bos_token='<s>':0, eos_token='</s>':2, sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>:1 
    def get_special_token_ids(self):
        return {"bos_token_id":0, "eos_token_id":2, "pad_token_id":1}
