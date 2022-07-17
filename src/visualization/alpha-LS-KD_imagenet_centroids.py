import torch
import numpy as np
import random
import argparse

from data_loader_imagenet import *
from torchvision.models import resnet50, resnet18

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Keep penultimate features as global varialble such that hook modifies these features
penultimate_fts = None


def get_penultimate_fts(self, input, output):
    global penultimate_fts
    penultimate_fts = output
    return None


def load_model_ls(alpha):
    """
    :return: model loaded with trained weights
    """

    # load model
    if args.loss == 'crossentropy':
        model = resnet50()
        path = f'./output/checkpoints/{args.model}-official.pth'
        ckpt = torch.load(path)  # 267
        model.load_state_dict(ckpt, strict=False)
    else:
        model = resnet50()
        path = f'./output/checkpoints/{args.model}-0.1.pth.tar'
        ckpt = torch.load(path)['state_dict']  # 320
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items() if k.startswith('module')}
        model.load_state_dict(ckpt)

    return model, ckpt


def load_model_kd(alpha, temperature):
    """
    :return: model loaded with trained weights
    """

    # load model
    if args.model == 'resnet18':
        model = resnet18()
    else:
        model = resnet50()
    path = f'./output/checkpoints/{args.model}-t=resnet50-a={alpha}-T={temperature}.pth.tar'
    ckpt = torch.load(path)['state_dict']  # 320
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items() if k.startswith('module')}
    model.load_state_dict(ckpt)

    return model, ckpt


def visualize(model, dataloader, num_sample=100, category=None):
    """
    :param dataloader: data_loader
    :return: visualize global features of train/valid/test samples
    """
    label_array = []
    feature_array = []
    model.cuda().eval()
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()

        # =================== ##### extract penultimate layer features =======================
        # Register hook to avg pool layer
        model.avgpool.register_forward_hook(get_penultimate_fts)
        with torch.no_grad():
            output = model(x)
            assert torch.is_tensor(penultimate_fts)
        feature = penultimate_fts.data.cpu().numpy().squeeze()
        one_vector = np.ones(shape=(feature.shape[0], 1))
        feature = np.concatenate((feature, one_vector), axis=1)
        label = y.data.cpu().numpy()
        feature_array.append(feature)
        label_array.append(label[:, np.newaxis])

    # todo: extract the specified number of samples per class
    output_array = np.concatenate(feature_array, axis=0)
    target_array = np.concatenate(label_array, axis=0)
    output_subset = []
    target_subset = []
    for i in category:
        sample_index = np.arange(num_sample)  # sample the same 100 samples of all cases
        tmp_index = np.where(target_array == i)[0][sample_index]  # sample 100 features from the same class
        output_subset.append(output_array[tmp_index])
        target_subset.append(target_array[tmp_index])
    output_subset_concat = np.concatenate(output_subset, axis=0)
    target_subset_concat = np.concatenate(target_subset, axis=0)
    print('Feature Shape :', output_subset_concat.shape)
    print('Target Shape :', target_subset_concat.shape)

    return output_subset_concat, target_subset_concat


def argparser():
    parser = argparse.ArgumentParser(description="Visualization of LS-KD features")
    parser.add_argument('--dataset', default='ImageNet')
    parser.add_argument('--model', default='resnet18', help='ModelType: {alexnet/resnet18|50|56}')
    parser.add_argument('--loss', default='crossentropy', help='crossentropy/labelsmoothing')
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--num_epoch', default=5)
    parser.add_argument('--num_sample', default=100, help='samples per class for visualization')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparser()

    # ---------------- Load data and class name -------------------- #
    CLASSES = ('miniature_poodle', 'standard_poodle', 'submarine')
    CLASSES_id = ('n02113712', 'n02113799', 'n04347754')
    CLASSES_idx = [266, 267, 833]
    # CLASSES = ['cabbage_butterfly', 'sulphur_butterfly', 'submarine']
    # CLASSES_id = ['n02280649', 'n02281406', 'n04347754']
    # CLASSES_idx = [324, 325, 833]
    category = [0, 1, 2]

    color = ['r', 'g', 'b']
    color_centroid = ['r', 'g', 'b']
    figure_idx = 1
    # load the corresponding weights index
    weights_idx = []
    for i in category:
        weights_idx.append(CLASSES_idx[i])

    train_loader, valid_loader = get_train_valid_loader(data_dir='../data/imagenet_visualization',
                                                        batch_size=args.batch_size, augment=True,
                                                        shuffle=False)

    # -------------------- model type choosing --------------------- #
    args.model = 'resnet18'
    if args.model == 'resnet18':
        temperature = ['1', '2']
    else:
        temperature = ['1', '3']
    alpha = ['0.0', '0.1']
    pca1 = PCA(n_components=2)
    pca2 = PCA(n_components=2)
    pca3 = PCA(n_components=2)
    pca4 = PCA(n_components=2)
    # ---------------- iteratively plot the features --------------- #
    # student resnet18 w/ KD and LS teacher (resnet50)
    for temp in temperature:  # temperature = 1/2
        for visual_set in [train_loader, valid_loader]:  # plot training / valid set features
            if visual_set == train_loader:
                set_name = 'Training'
            else:
                set_name = 'Validation'
            for a in alpha:  # \alpha =0.0 / 0.1
                # loss_name = r'$\alpha$' + '={}'.format(alpha[i])
                if a == '0.0':
                    loss_name = ' w/ KD' + r' $T$={} '.format(temp) + '\n' + 'Teacher w/o LS'
                else:
                    loss_name = ' w/ KD' + r' $T$={} '.format(temp) + '\n' + 'Teacher w/ LS (' + r'$\alpha$' + '={})'.format(a)
                model, state = load_model_kd(alpha=a, temperature=temp)

                # ------- Step 1. Compute the orthonormal basis------- #
                if args.model == 'resnet18' or args.model == 'resnet50':
                    classifier_weight = state['fc.weight']
                    classifier_bias = state['fc.bias']
                else:
                    classifier_weight = None
                    classifier_bias = None

                assert torch.is_tensor(classifier_weight)
                assert torch.is_tensor(classifier_bias)
                weights_value = classifier_weight.data.cpu().numpy()[weights_idx, :]  # (3, *)
                bias_value = classifier_bias.data.cpu().numpy()
                bias_value = bias_value.reshape(bias_value.size, 1)[category, :]  # dim = (3, 1)
                weights = np.concatenate((weights_value, bias_value), axis=1)  # (3, * + 1)
                basis, _ = np.linalg.qr(weights.T)  # (* + 1, 3)

                # ------------ Step 2. Feature Extraction ------------ #
                args.num_sample = 150
                if visual_set == valid_loader:
                    args.num_sample = 50
                output_feature, output_target = visualize(model=model, dataloader=visual_set,
                                                          num_sample=args.num_sample,
                                                          category=category)  # (450, 2048)

                # ------- Step 3. Projection into new 3-D subspace -------- #
                # find the centroids of features:
                centroids = []
                for k in range(len(category)):
                    centroids.append(output_feature[k * args.num_sample:(k + 1) * args.num_sample, :].mean(axis=0))
                centroids = np.array(centroids)
                output_project = np.dot(output_feature, basis)  # dim = (300, 3)
                centroids_project = np.dot(centroids, basis)  # dim = (3, 3)
                feature_concat = np.concatenate((output_project, centroids_project), axis=0)

                # ------- Step 4. Dimension reduction from 3-D to 2-D -------- #
                output_array = None
                centroids_array = None
                if a == '0.0' and temp == temperature[0]:
                    if set_name == 'Training':
                        pca1.fit(feature_concat)
                        output_array = pca1.transform(output_project)
                        centroids_array = pca1.transform(centroids_project)
                    else:
                        output_array = pca1.transform(output_project)
                        centroids_array = pca1.transform(centroids_project)
                elif a == '0.1' and temp == temperature[0]:
                    if set_name == 'Training':
                        pca2.fit(feature_concat)
                        output_array = pca2.transform(output_project)
                        centroids_array = pca2.transform(centroids_project)
                    else:
                        output_array = pca2.transform(output_project)
                        centroids_array = pca2.transform(centroids_project)
                elif a == '0.0' and temp == temperature[1]:
                    if set_name == 'Training':
                        pca3.fit(feature_concat)
                        output_array = pca3.transform(output_project)
                        centroids_array = pca3.transform(centroids_project)
                    else:
                        output_array = pca3.transform(output_project)
                        centroids_array = pca3.transform(centroids_project)
                elif a == '0.1' and temp == temperature[1]:
                    if set_name == 'Training':
                        pca4.fit(feature_concat)
                        output_array = pca4.transform(output_project)
                        centroids_array = pca4.transform(centroids_project)
                    else:
                        output_array = pca4.transform(output_project)
                        centroids_array = pca4.transform(centroids_project)
                else:
                    pass
                print('feature shape after dimension reduction', output_array.shape)  # (300, 2)

                # ------- Step 5. Draw a subplot  -------- #
                plt.rcParams['figure.figsize'] = 40, 20
                plt.rcParams['font.size'] = 35
                plt.rcParams["font.family"] = "Times New Roman"
                # plt.rcParams['axes.xmargin'] = 0
                plt.tight_layout()
                plt.subplot(2, 4, figure_idx)  # 8 subplots in total

                # plot the features
                for j, subclass in enumerate(category):
                    plt.scatter(output_array[j * args.num_sample:(j + 1) * args.num_sample, 0],
                                output_array[j * args.num_sample:(j + 1) * args.num_sample, 1],
                                alpha=0.6, c=color[j], s=300, label=CLASSES[category[j]])
                plt.legend(prop={'size': 25})

                # plot the centroids
                for j, subclass in enumerate(category):
                    plt.scatter(centroids_array[j:(j + 1), 0],
                                centroids_array[j:(j + 1), 1],
                                alpha=0.95, c=color_centroid[j], marker='*', s=1500)

                # plot setting
                if args.model == 'resnet18':
                    title = f'ResNet18 {set_name}' + f'{loss_name}'
                else:
                    title = f'ResNet50 {set_name}' + f'{loss_name}'
                plt.title(title)
                font = {'size': 35}
                # if figure_idx == 1 or figure_idx == 5:
                #     plt.ylabel(args.dataset, fontdict=font)
                figure_idx += 1
    plt.savefig(f'ls_kd_imagenet_{CLASSES[0]}_{CLASSES[1]}_{CLASSES[2]}.pdf', format='pdf', dpi=1200, bbox_inches='tight', pad_inches=0)
    plt.show()
