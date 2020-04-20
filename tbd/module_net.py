# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import torch
import torch.nn as nn
from copy import copy
from . import modules
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class TbDNet(nn.Module):
    """ The real deal. A full Transparency by Design network (TbD-net).

    Extended Summary
    ----------------
    A :class:`TbDNet` holds neural :mod:`modules`, a stem network, and a classifier network. It
    hooks these all together to answer a question given some scene and a program describing how to
    arrange the neural modules.
    """

    def __init__(self,
                 vocab,
                 feature_dim=(512, 28, 28),
                 module_dim=128,
                 cls_proj_dim=512,
                 fc_dim=1024):
        """ Initializes a TbDNet object.

        Parameters
        ----------
        vocab : Dict[str, Dict[Any, Any]]
            The vocabulary holds dictionaries that provide handles to various objects. Valid keys
            into vocab are
            - 'answer_idx_to_token' whose keys are ints and values strings
            - 'answer_token_to_idx' whose keys are strings and values ints
            - 'program_idx_to_token' whose keys are ints and values strings
            - 'program_token_to_idx' whose keys are strings and values ints
            These value dictionaries provide retrieval of an answer word or program token from an
            index, or an index from a word or program token.

        feature_dim : the tuple (K, R, C), optional
            The shape of input feature tensors, excluding the batch size.

        module_dim : int, optional
            The depth of each neural module's convolutional blocks.

        cls_proj_dim : int, optional
            The depth to project the final feature map to before classification.
        """
        super().__init__()

        # The stem takes features from ResNet (or another feature extractor) and projects down to
        # a lower-dimensional space for sending through the TbD-net
        self.stem = nn.Sequential(nn.Conv2d(feature_dim[0], module_dim, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(module_dim, module_dim, kernel_size=3, padding=1),
                                  nn.ReLU()
                                  )

        module_rows, module_cols = feature_dim[1], feature_dim[2]

        # The classifier takes the output of the last module (which will be a Query or Equal module)
        # and produces a distribution over answers
        self.classifier = nn.Sequential(nn.Conv2d(module_dim, cls_proj_dim, kernel_size=1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        Flatten(),
                                        nn.Linear(cls_proj_dim * module_rows * module_cols // 4,
                                                  fc_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(fc_dim, 28)  # note no softmax here
                                        )

        self.function_modules = {}  # holds our modules
        self.vocab = vocab
        # go through the vocab and add all the modules to our model
        for module_name in vocab['program_token_to_idx']:
            if module_name in ['<NULL>', '<START>', '<END>', '<UNK>', 'unique']:
                continue  # we don't need modules for the placeholders

            # figure out which module we want we use
            if module_name == 'scene':
                # scene is just a flag that indicates the start of a new line of reasoning
                # we set `module` to `None` because we still need the flag 'scene' in forward()
                module = None
            elif module_name == 'intersect':
                module = modules.AndModule()
            elif module_name == 'union':
                module = modules.OrModule()
            elif 'equal' in module_name or module_name in {'less_than', 'greater_than'}:
                module = modules.ComparisonModule(module_dim)
            elif 'query' in module_name or module_name in {'exist', 'count'}:
                module = modules.QueryModule(module_dim)
            elif 'relate' in module_name:
                if 'relate' not in self.function_modules:
                    m = modules.RelateModuleBase(module_dim)
                    self.function_modules['relate'] = m

                module = modules.RelateModuleDerived(module_dim, self.function_modules['relate'])
            elif 'same' in module_name:
                module = modules.SameModule(module_dim)
            else:
                if 'filter' not in self.function_modules:
                    m = modules.AttentionModuleBase(module_dim)
                    self.function_modules['filter'] = m

                module = modules.AttentionModuleDerived(module_dim, self.function_modules['filter'])

            # add the module to our dictionary and register its parameters so it can learn
            self.function_modules[module_name] = module
            self.add_module(module_name, module)

        # this is used as input to the first AttentionModule in each program
        ones = torch.ones(1, 1, module_rows, module_cols)
        self.ones_var = ones.cuda() if torch.cuda.is_available() else ones

        self._attention_sum = 0

    @property
    def attention_sum(self):
        '''
        Returns
        -------
        attention_sum : int
            The sum of attention masks produced during the previous forward pass, or zero if a
            forward pass has not yet happened.

        Extended Summary
        ----------------
        This property holds the sum of attention masks produced during a forward pass of the model.
        It will hold the sum of all the AttentionModule, RelateModule, and SameModule outputs. This
        can be used to regularize the output attention masks, hinting to the model that spurious
        activations that do not correspond to objects of interest (e.g. activations in the
        background) should be minimized. For example, a small factor multiplied by this could be
        added to your loss function to add this type of regularization as in:

            loss = xent_loss(outs, answers)
            loss += executor.attention_sum * 2.5e-07
            loss.backward()

        where `xent_loss` is our loss function, `outs` is the output of the model, `answers` is the
        PyTorch `Tensor` containing the answers, and `executor` is this model. The above block
        will penalize the model's attention outputs multiplied by a factor of 2.5e-07 to push the
        model to produce sensible, minimal activations.
        '''
        return self._attention_sum

    def level_tree(self, program):
        dic = dict()
        secondArg = -1
        depth = 0
        secondDepth = 0
        last = -1
        for idx in range(len(program) - 1, -1, -1):
            i = program[idx].item()
            module_type = self.vocab['program_idx_to_token'][i]
            if module_type in {'<NULL>', '<START>', '<END>', '<UNK>', 'unique'}:
                continue
            elif module_type == 'scene':
                # store the previous output; it will be needed later
                # scene is just a flag, performing no computation
                secondArg = last
                last = -1
                secondDepth = depth
                depth = 0
                # if idx == end -1:
                #   secondArg = None
                #   depth = secondDepth
                #   secondDepth = 0
                continue
            elif 'equal' in module_type or module_type in {'intersect', 'union', 'less_than',
                                                         'greater_than'}:
                dic[idx] = {'function': module_type, 'input1': last, 'input2': secondArg, 'level': None}
                # secondArg = None
                depth = 1 + max(depth, secondDepth)
                secondDepth = 0
                secondArg = -1
                last = idx
            else:
                # these modules take extracted image features and a previous attention
                depth += 1
                dic[idx] = {'function': module_type, 'input1': -2, 'input2': last, 'level': None}
                last = idx
        levels = [1]
        for i in range(depth):
            newlevels = []
            for j in levels:
                try:
                    dic[j]['level'] = depth - i
                    if dic[j]['input1'] > 0:
                        newlevels.append(dic[j]['input1'])
                    if dic[j]['input2'] > 0:
                        newlevels.append(dic[j]['input2'])
                except:
                    pass
            levels = newlevels
        return dic

    def getLvlToIdx(self, tree):
        dic = dict()
        for i in tree.keys():
            try:
                dic[tree[i]['level']].append(i)
            except:
                dic[tree[i]['level']] = [i]
        return dic, max(dic.keys())

    # Think of 'levels' as Time
    def forward(self, feats, programs):
        batch_size = feats.size(0)
        assert batch_size == len(programs)

        feat_input_volume = self.stem(feats)  # forward all the features through the stem at once
        outputs_of_modules = []
        levels = []
        level_to_idxs = []
        maxLvls = []
        maxLevel = -1
        for i in range(batch_size):
            tmpLst = []
            for j in range(programs[0].shape[0]):  # Programs are padded to fixed length
                tmpLst.append(None)
            tmpLst.append(feat_input_volume[i:i+1])
            tmpLst.append(self.ones_var)

            outputs_of_modules.append(tmpLst)
            tmp = self.level_tree(programs[i].data.cpu().numpy())
            levels.append(tmp)
            tmp = self.getLvlToIdx(tmp)
            level_to_idxs.append(tmp[0])
            maxLvls.append(tmp[1])
            maxLevel = max(maxLevel, tmp[1])

        # for t in range(1, maxLevel+1):

        # We compose each module network individually since they are constructed on a per-question
        # basis. Here we go through each program in the batch, construct a modular network based on
        # it, and send the image forward through the modular structure. We keep the output of the
        # last module for each program in final_module_outputs. These are needed to then compute a
        # distribution over answers for all the questions as a batch.
        final_module_outputs = []
        for t in range(1, maxLevel+1):
            functionToBatch = dict()
            scheduledToExec = dict()
            input1ToExec = dict()
            input2ToExec = dict()    
            for n in range(batch_size):
                if t>maxLvls[n]:
                    continue
                feat_input = feat_input_volume[n:n + 1]
                for node in level_to_idxs[n][t]:
                    try:
                        scheduledToExec[levels[n][node]['function']].append(node)
                        input1ToExec[levels[n][node]['function']].append(levels[n][node]['input1'])
                        input2ToExec[levels[n][node]['function']].append(levels[n][node]['input2'])
                        functionToBatch[levels[n][node]['function']].append(n)
                    except:
                        scheduledToExec[levels[n][node]['function']] = [node]
                        input1ToExec[levels[n][node]['function']] = [levels[n][node]['input1']]
                        input2ToExec[levels[n][node]['function']] = [levels[n][node]['input2']]
                        functionToBatch[levels[n][node]['function']] = [n]
                        
            for function in scheduledToExec.keys():
                module = self.function_modules[function]
                tmp1 = []
                tmp2 = []
                for indx in range(len(input1ToExec[function])):
                    tmp1.append(outputs_of_modules[functionToBatch[function][indx]][input1ToExec[function][indx]])
                    tmp2.append(outputs_of_modules[functionToBatch[function][indx]][input2ToExec[function][indx]])
                      
                inputs1 = torch.cat(tmp1, 0)
                inputs2 = torch.cat(tmp2, 0)
#                 print(inputs1.shape, inputs2.shape)
                out = module(inputs1, inputs2)
                indices = scheduledToExec[function]
                for out_idx, output_idx in enumerate(indices):
                    outputs_of_modules[functionToBatch[function][out_idx]][output_idx] = out[out_idx:out_idx+1]
        
        final_module_outputs = [outputs_of_modules[i][1] for i in range(batch_size)]
        final_module_outputs = torch.cat(final_module_outputs, 0)
        return self.classifier(final_module_outputs)

    def forward_and_return_intermediates(self, program_var, feats_var):
        """ Forward program `program_var` and image features `feats_var` through the TbD-Net
        and return an answer and intermediate outputs.

        Parameters
        ----------
        program_var : torch.Tensor
            The program to carry out.

        feats_var : torch.Tensor
            The image features to operate on.

        Returns
        -------
        Tuple[str, List[Tuple[str, numpy.ndarray]]]
            A tuple of (answer, [(operation, attention), ...]). Note that some of the
            intermediates will be `None`, which indicates a break in the logic chain. For
            example, in the question:
                "What color is the cube to the left of the sphere and right of the cylinder?"
            We have 3 distinct chains of reasoning. We first localize the sphere and look left. We
            then localize the cylinder and look right. Thirdly, we look at the intersection of these
            two, and find the cube.
        """
        intermediaries = []
        # the logic here is the same as self.forward()
        scene_input = self.stem(feats_var)
        output = scene_input
        saved_output = None
        for i in reversed(program_var.data.cpu().numpy()[0]):
            module_type = self.vocab['program_idx_to_token'][i]
            if module_type in {'<NULL>', '<START>', '<END>', '<UNK>', 'unique'}:
                continue

            module = self.function_modules[module_type]
            if module_type == 'scene':
                saved_output = output
                output = self.ones_var
                intermediaries.append(None)  # indicates a break/start of a new logic chain
                continue

            if 'equal' in module_type or module_type in {'intersect', 'union', 'less_than',
                                                         'greater_than'}:
                output = module(output, saved_output)
            else:
                output = module(scene_input, output)

            if module_type in {'intersect', 'union'}:
                intermediaries.append(None)  # this is the start of a new logic chain

            if module_type in {'intersect', 'union'} or any(s in module_type for s in ['same',
                                                                                       'filter',
                                                                                       'relate']):
                intermediaries.append((module_type, output.data.cpu().numpy().squeeze()))

        _, pred = self.classifier(output).max(1)
        return (self.vocab['answer_idx_to_token'][pred.item()], intermediaries)


def load_tbd_net(checkpoint, vocab):
    """ Convenience function to load a TbD-Net model from a checkpoint file.

    Parameters
    ----------
    checkpoint : Union[pathlib.Path, str]
        The path to the checkpoint.

    vocab : Dict[str, Dict[any, any]]
        The vocabulary file associated with the TbD-Net. For an extended description, see above.

    Returns
    -------
    torch.nn.Module
        The TbD-Net model.

    Notes
    -----
    This pushes the TbD-Net model to the GPU if a GPU is available.
    """
    tbd_net = TbDNet(vocab)
    tbd_net.load_state_dict(torch.load(str(checkpoint), map_location={'cuda:0': 'cpu'}))
    if torch.cuda.is_available():
        tbd_net.cuda()
    return tbd_net
