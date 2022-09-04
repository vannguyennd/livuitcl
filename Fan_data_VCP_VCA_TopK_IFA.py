import os
import numpy as np
import argparse


def get_selected_lines(lines, args):
    selected_line = []
    find_sl = False
    k_selected_line = 0
    for i, id_line in enumerate(lines):
        if id_line.__contains__('are all selected'):
            ji = i + 1
            while ji < len(lines) and k_selected_line <= args.k_select:
                if lines[ji].__contains__('are ground truth'):
                    break
                if lines[ji].strip() != '':
                    selected_line.append(lines[ji])
                    k_selected_line += 1
                ji = ji + 1
            find_sl = True
            break
    if find_sl is False:
        print(lines[0])
        print('cannot finding selected statements')
    return selected_line


def get_ground_truth_lines(lines):
    selected_line = []
    find_sl = False
    for i, id_line in enumerate(lines):
        if id_line.__contains__('are ground truth'):
            ji = i + 1
            while ji < len(lines):
                selected_line.append(lines[ji].strip())
                ji = ji + 1
            find_sl = True
            break
    if find_sl is False:
        print(lines[0])
        print('cannot finding the ground truth statements')
    return selected_line


def compare_equal(p_item_a, p_item_b):
    string_a = ''
    string_b = ''

    for item in p_item_a:
        string_a += item
    string_a = string_a.strip().lower()

    for item in p_item_b:
        string_b += item
    string_b = string_b.strip().lower()

    if string_a == string_b:
        return True
    else:
        return False


def in_list(item, item_ls):
    for i_item in item_ls:
        if compare_equal(item.split(' '), i_item.split(' ')):
            return True
    return False


def check_in_consideration(gt_lines, all_lines, args):
    original_lines = []
    find_sl = False
    for i, id_line in enumerate(all_lines):
        if id_line.__contains__('are original function'):
            ji = i + 1
            while 'are all selected' not in all_lines[ji]:
                original_lines.append(all_lines[ji])
                ji = ji + 1
            find_sl = True
            break
    if find_sl is False:
        print(all_lines[0])
        print('cannot finding selected statements')
    else:
        original_lines = original_lines[:args.max_statements]

    considered_gt = []
    for i in range(len(gt_lines)):
        if in_list(gt_lines[i], original_lines):
            considered_gt.append(gt_lines[i])

    return considered_gt


def compute_VCP(i_list_folders, i_folder, result_file, all_folders, args):
    for ij_folder in i_list_folders:
        if ij_folder.__contains__('vul'):
            print(ij_folder)

            v_sg_cor_line = 0
            v_gt_in_cor_line = 0

            t_sg_cor_line = 0
            t_gt_in_cor_line = 0

            ij_list_files = os.listdir(os.path.join(all_folders, i_folder, ij_folder))
            ij_list_files = sorted(ij_list_files, key=lambda file: int(file.split('.')[0]))

            for idx, ij_file in enumerate(ij_list_files):
                if idx < len(ij_list_files) // 2:
                    with open(os.path.join(all_folders, i_folder, ij_folder, ij_file)) as f:
                        all_lines_st = []
                        all_lines = f.readlines()

                    for line in all_lines:
                        all_lines_st.append(line.strip().lower())

                    selected_lines = get_selected_lines(all_lines_st, args)
                    selected_lines_lf = []
                    for i_sl in selected_lines:
                        if in_list(i_sl, selected_lines_lf) is False:
                            selected_lines_lf.append(i_sl)

                    ground_truth_lines = get_ground_truth_lines(all_lines_st)
                    considered_gt_lines = check_in_consideration(ground_truth_lines, all_lines_st, args)
                    ground_truth_lines = list(dict.fromkeys(considered_gt_lines))

                    inter_sl_gt = []
                    for i_sl_line in selected_lines_lf:
                        if in_list(i_sl_line, ground_truth_lines):
                            inter_sl_gt.append(i_sl_line)

                    if len(ground_truth_lines) != 0:
                        v_sg_cor_line += len(inter_sl_gt)
                    v_gt_in_cor_line += len(ground_truth_lines)
                else:
                    with open(os.path.join(all_folders, i_folder, ij_folder, ij_file)) as f:
                        all_lines_st = []
                        all_lines = f.readlines()

                    for line in all_lines:
                        all_lines_st.append(line.strip().lower())

                    selected_lines = get_selected_lines(all_lines_st, args)
                    selected_lines_lf = []
                    for i_sl in selected_lines:
                        if in_list(i_sl, selected_lines_lf) is False:
                            selected_lines_lf.append(i_sl)

                    ground_truth_lines = get_ground_truth_lines(all_lines_st)
                    considered_gt_lines = check_in_consideration(ground_truth_lines, all_lines_st, args)
                    ground_truth_lines = list(dict.fromkeys(considered_gt_lines))

                    inter_sl_gt = []
                    for i_sl_line in selected_lines_lf:
                        if in_list(i_sl_line, ground_truth_lines):
                            inter_sl_gt.append(i_sl_line)

                    if len(ground_truth_lines) != 0:
                        t_sg_cor_line += len(inter_sl_gt)
                    t_gt_in_cor_line += len(ground_truth_lines)

            v_gt_all_line = v_gt_in_cor_line
            t_gt_all_line = t_gt_in_cor_line

            result_file.write('gt_val_VCP: ' + str(np.round(1.0 * v_sg_cor_line / v_gt_all_line, 5)) + '\t\t\t')
            result_file.write('gt_test_VCP: ' + str(np.round(1.0 * t_sg_cor_line / t_gt_all_line, 5)) + '\t\t\t')
            result_file.write('gt_val_test_VCP: ' + str(np.round(1.0 * (v_sg_cor_line+t_sg_cor_line) / (v_gt_all_line+t_gt_all_line), 5)) + '\n')


def compute_VCA(i_list_folders, i_folder, result_file, all_folders, args):
    for ij_folder in i_list_folders:
        if ij_folder.__contains__('vul'):
            print(ij_folder)

            v_sg_cor_line = 0
            v_gt_in_cor_line = 0

            t_sg_cor_line = 0
            t_gt_in_cor_line = 0

            ij_list_files = os.listdir(os.path.join(all_folders, i_folder, ij_folder))
            ij_list_files = sorted(ij_list_files, key=lambda file: int(file.split('.')[0]))

            for idx, ij_file in enumerate(ij_list_files):
                if idx < len(ij_list_files) // 2:
                    with open(os.path.join(all_folders, i_folder, ij_folder, ij_file)) as f:
                        all_lines_st = []
                        all_lines = f.readlines()

                    for line in all_lines:
                        all_lines_st.append(line.strip().lower())

                    selected_lines = get_selected_lines(all_lines_st, args)
                    selected_lines_lf = []
                    for i_sl in selected_lines:
                        if in_list(i_sl, selected_lines_lf) is False:
                            selected_lines_lf.append(i_sl)

                    ground_truth_lines = get_ground_truth_lines(all_lines_st)
                    considered_gt_lines = check_in_consideration(ground_truth_lines, all_lines_st, args)
                    ground_truth_lines = list(dict.fromkeys(considered_gt_lines))

                    inter_sl_gt = []
                    for i_sl_line in selected_lines_lf:
                        if in_list(i_sl_line, ground_truth_lines):
                            inter_sl_gt.append(i_sl_line)

                    if len(ground_truth_lines) != 0:
                        if len(inter_sl_gt) == len(ground_truth_lines):
                            v_sg_cor_line += 1
                        v_gt_in_cor_line += 1
                else:
                    with open(os.path.join(all_folders, i_folder, ij_folder, ij_file)) as f:
                        all_lines_st = []
                        all_lines = f.readlines()

                    for line in all_lines:
                        all_lines_st.append(line.strip().lower())

                    selected_lines = get_selected_lines(all_lines_st, args)
                    selected_lines_lf = []
                    for i_sl in selected_lines:
                        if in_list(i_sl, selected_lines_lf) is False:
                            selected_lines_lf.append(i_sl)

                    ground_truth_lines = get_ground_truth_lines(all_lines_st)
                    considered_gt_lines = check_in_consideration(ground_truth_lines, all_lines_st, args)
                    ground_truth_lines = list(dict.fromkeys(considered_gt_lines))

                    inter_sl_gt = []
                    for i_sl_line in selected_lines_lf:
                        if in_list(i_sl_line, ground_truth_lines):
                            inter_sl_gt.append(i_sl_line)

                    if len(ground_truth_lines) != 0:
                        if len(inter_sl_gt) == len(ground_truth_lines):
                            t_sg_cor_line += 1
                        t_gt_in_cor_line += 1

            v_gt_all_line = v_gt_in_cor_line
            t_gt_all_line = t_gt_in_cor_line

            result_file.write('gt_val_VCA: ' + str(np.round(1.0 * v_sg_cor_line / v_gt_all_line, 5)) + '\t\t\t')
            result_file.write('gt_test_VCA: ' + str(np.round(1.0 * t_sg_cor_line / t_gt_all_line, 5)) + '\t\t\t')
            result_file.write('gt_val_test_VCA: ' + str(np.round(1.0 * (v_sg_cor_line+t_sg_cor_line) / (v_gt_all_line+t_gt_all_line), 5)) + '\n')


def compute_TOPK_ACC(i_list_folders, i_folder, result_file, all_folders, args):
    for ij_folder in i_list_folders:
        if ij_folder.__contains__('vul'):
            print(ij_folder)

            v_sg_cor_line = 0
            v_gt_in_cor_line = 0

            t_sg_cor_line = 0
            t_gt_in_cor_line = 0

            ij_list_files = os.listdir(os.path.join(all_folders, i_folder, ij_folder))
            ij_list_files = sorted(ij_list_files, key=lambda file: int(file.split('.')[0]))

            for idx, ij_file in enumerate(ij_list_files):
                if idx < len(ij_list_files) // 2:
                    with open(os.path.join(all_folders, i_folder, ij_folder, ij_file)) as f:
                        all_lines_st = []
                        all_lines = f.readlines()

                    for line in all_lines:
                        all_lines_st.append(line.strip().lower())

                    selected_lines = get_selected_lines(all_lines_st, args)
                    selected_lines_lf = []
                    for i_sl in selected_lines:
                        if in_list(i_sl, selected_lines_lf) is False:
                            selected_lines_lf.append(i_sl)

                    ground_truth_lines = get_ground_truth_lines(all_lines_st)
                    considered_gt_lines = check_in_consideration(ground_truth_lines, all_lines_st, args)
                    ground_truth_lines = list(dict.fromkeys(considered_gt_lines))

                    inter_sl_gt = False
                    for i_sl_line in selected_lines_lf:
                        if in_list(i_sl_line, ground_truth_lines):
                            inter_sl_gt = True
                            break

                    if len(ground_truth_lines) != 0:
                        if inter_sl_gt == True:
                            v_sg_cor_line += 1
                        v_gt_in_cor_line += 1
                else:
                    with open(os.path.join(all_folders, i_folder, ij_folder, ij_file)) as f:
                        all_lines_st = []
                        all_lines = f.readlines()

                    for line in all_lines:
                        all_lines_st.append(line.strip().lower())

                    selected_lines = get_selected_lines(all_lines_st, args)
                    selected_lines_lf = []
                    for i_sl in selected_lines:
                        if in_list(i_sl, selected_lines_lf) is False:
                            selected_lines_lf.append(i_sl)

                    ground_truth_lines = get_ground_truth_lines(all_lines_st)
                    considered_gt_lines = check_in_consideration(ground_truth_lines, all_lines_st, args)
                    ground_truth_lines = list(dict.fromkeys(considered_gt_lines))

                    inter_sl_gt = False
                    for i_sl_line in selected_lines_lf:
                        if in_list(i_sl_line, ground_truth_lines):
                            inter_sl_gt = True
                            break

                    if len(ground_truth_lines) != 0:
                        if inter_sl_gt == True:
                            t_sg_cor_line += 1
                        t_gt_in_cor_line += 1

            v_gt_all_line = v_gt_in_cor_line
            t_gt_all_line = t_gt_in_cor_line

            result_file.write('gt_val_top_' + str(args.k_select) + ': ' + str(np.round(1.0 * v_sg_cor_line / v_gt_all_line, 5)) + '\t\t')
            result_file.write('gt_test_top_' + str(args.k_select) + ': ' + str(np.round(1.0 * t_sg_cor_line / t_gt_all_line, 5)) + '\t\t\t')
            result_file.write('gt_val_test_top_' + str(args.k_select) + ': ' + str(np.round(1.0 * (v_sg_cor_line+t_sg_cor_line) / (v_gt_all_line+t_gt_all_line), 5)) + '\n')


def compute_IFA(i_list_folders, i_folder, result_file, all_folders, args):
    for ij_folder in i_list_folders:
        if ij_folder.__contains__('vul'):
            print(ij_folder)

            v_sg_cor_line = 0
            v_gt_in_cor_line = 0

            t_sg_cor_line = 0
            t_gt_in_cor_line = 0

            ij_list_files = os.listdir(os.path.join(all_folders, i_folder, ij_folder))
            ij_list_files = sorted(ij_list_files, key=lambda file: int(file.split('.')[0]))

            for idx, ij_file in enumerate(ij_list_files):
                if idx < len(ij_list_files) // 2:
                    with open(os.path.join(all_folders, i_folder, ij_folder, ij_file)) as f:
                        all_lines_st = []
                        all_lines = f.readlines()

                    for line in all_lines:
                        all_lines_st.append(line.strip().lower())

                    selected_lines = get_selected_lines(all_lines_st, args)
                    selected_lines_lf = []
                    for i_sl in selected_lines:
                        if in_list(i_sl, selected_lines_lf) is False:
                            selected_lines_lf.append(i_sl)

                    ground_truth_lines = get_ground_truth_lines(all_lines_st)
                    considered_gt_lines = check_in_consideration(ground_truth_lines, all_lines_st, args)
                    ground_truth_lines = list(dict.fromkeys(considered_gt_lines))

                    inter_sl_gt = 0
                    gt_exist_in_selected = False
                    for isl_idx, i_sl_line in enumerate(selected_lines_lf):
                        if in_list(i_sl_line, ground_truth_lines):
                            inter_sl_gt = isl_idx
                            gt_exist_in_selected = True
                            break

                    if len(ground_truth_lines) != 0:
                        if gt_exist_in_selected == True:
                            v_sg_cor_line += inter_sl_gt
                        v_gt_in_cor_line += 1
                else:
                    with open(os.path.join(all_folders, i_folder, ij_folder, ij_file)) as f:
                        all_lines_st = []
                        all_lines = f.readlines()

                    for line in all_lines:
                        all_lines_st.append(line.strip().lower())

                    selected_lines = get_selected_lines(all_lines_st, args)
                    selected_lines_lf = []
                    for i_sl in selected_lines:
                        if in_list(i_sl, selected_lines_lf) is False:
                            selected_lines_lf.append(i_sl)

                    ground_truth_lines = get_ground_truth_lines(all_lines_st)
                    considered_gt_lines = check_in_consideration(ground_truth_lines, all_lines_st, args)
                    ground_truth_lines = list(dict.fromkeys(considered_gt_lines))

                    inter_sl_gt = 0
                    gt_exist_in_selected = False
                    for isl_idx, i_sl_line in enumerate(selected_lines_lf):
                        if in_list(i_sl_line, ground_truth_lines):
                            inter_sl_gt = isl_idx
                            gt_exist_in_selected = True
                            break

                    if len(ground_truth_lines) != 0:
                        if gt_exist_in_selected == True:
                            t_sg_cor_line += inter_sl_gt
                        t_gt_in_cor_line += 1

            v_gt_all_line = v_gt_in_cor_line
            t_gt_all_line = t_gt_in_cor_line

            result_file.write('gt_val_IFA: ' + str(np.round(1.0 * v_sg_cor_line / v_gt_all_line, 5)) + '\t\t\t')
            result_file.write('gt_test_IFA: ' + str(np.round(1.0 * t_sg_cor_line / t_gt_all_line, 5)) + '\t\t\t')
            result_file.write('gt_val_test_IFA: ' + str(np.round(1.0 * (v_sg_cor_line+t_sg_cor_line) / (v_gt_all_line+t_gt_all_line), 5)) + '\n')


def main():
    """
    ===
    for example
    python Fan_data_VCP_VCA_TopK_IFA.py --home_dir=./Fan_data_results/
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--home_dir", default='./Fan_data_results/', type=str, help="The home directory used in the evaluation process.")
    parser.add_argument('--max_statements', type=int, default=150, help="The number of considered statements for each function.")
    parser.add_argument('--k_select', type=int, default=10, help="The number of selected statements.")

    args = parser.parse_args()

    all_folders_dir_save = args.home_dir
    all_folders = args.home_dir + 'saved_models' +  '/'
    all_folders_dir_rs = args.home_dir + 'history_logs_load_predictions' + '/'

    list_folders = os.listdir(all_folders)
    list_folders = [f.lower() for f in list_folders]

    result_file = open(all_folders_dir_save + 'VCP_VCA_TopK_IFA_' + str(args.k_select) + '.txt', 'w')

    all_lines_rs_ls = []
    with open(os.path.join(all_folders_dir_rs, 'his_load_predictions.txt')) as f_rs:
        all_lines_rs = f_rs.readlines()
        for idx, i_line in enumerate(all_lines_rs):
            all_lines_rs_ls.append(i_line)

    for i_folder in list_folders:
        print(i_folder)

        i_list_folders = os.listdir(os.path.join(all_folders, i_folder))
        i_list_folders = [f.lower() for f in i_list_folders if '_result_' in f]

        for idx_il, i_line in enumerate(all_lines_rs_ls):
            if i_line.__contains__(' --- '):
                i_line_e = i_line.strip().split(' --- ')
                i_l_folders = i_folder.split('_')
                if np.float(i_l_folders[0]) == np.float(i_line_e[0].split(': ')[1]) and \
                                np.float(i_l_folders[1]) == np.float(i_line_e[1].split(': ')[1]) and \
                                np.float(i_l_folders[2]) == np.float(i_line_e[2].split(': ')[1]) and \
                                np.float(i_l_folders[3]) == np.float(i_line_e[3].split(': ')[1]) and \
                                np.float(i_l_folders[4]) == np.float(i_line_e[4].split(': ')[1]) and \
                                np.float(i_l_folders[5]) == np.float(i_line_e[5].split(': ')[1]):
                    result_file.write(i_line)
                    result_file.write(all_lines_rs_ls[idx_il+1])
                    break

        compute_VCP(i_list_folders, i_folder, result_file, all_folders, args)
        compute_VCA(i_list_folders, i_folder, result_file, all_folders, args)
        compute_TOPK_ACC(i_list_folders, i_folder, result_file, all_folders, args)
        compute_IFA(i_list_folders, i_folder, result_file, all_folders, args)


if __name__ == "__main__":
    main()
