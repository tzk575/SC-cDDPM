
import torch





def calculate_overlap(airfoil):
    batch_size, n_points, _ = airfoil.size()


    y_upper = airfoil[:, :, 0]
    y_lower = airfoil[:, :, 1]

    delta_y = y_lower - y_upper
    regular = torch.maximum(delta_y, torch.tensor(0.0))

    print('the airfoils overlap as following')
    for i in range(batch_size):
        indices = torch.nonzero(delta_y[i] >= 0, as_tuple=True)[0]
        print(f'For airfoil {i}, the overlapping indices are: {indices.tolist()}')



    overlap_regular = torch.sum(regular, dim=1, keepdim=True)
    overlap_loss = torch.mean(overlap_regular)


    return overlap_loss




