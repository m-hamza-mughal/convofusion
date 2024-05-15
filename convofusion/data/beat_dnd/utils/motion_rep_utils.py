# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This source code is modified for the purpose of the project specific to the 
# dataloader and the evaluation of the model. The original source code is
# available at https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html

from typing import Optional

import torch
import torch.nn.functional as F

import numpy as np

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])



def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


# def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
#     """
#     Convert rotations given as axis/angle to rotation matrices.

#     Args:
#         axis_angle: Rotations given as a vector in axis angle form,
#             as a tensor of shape (..., 3), where the magnitude is
#             the angle turned anticlockwise in radians around the
#             vector's direction.

#     Returns:
#         Rotation matrices as tensor of shape (..., 3, 3).
#     """
#     return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))



# [docs]def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
#     """
#     Convert rotations given as rotation matrices to axis/angle.

#     Args:
#         matrix: Rotation matrices as tensor of shape (..., 3, 3).

#     Returns:
#         Rotations given as a vector in axis angle form, as a tensor
#             of shape (..., 3), where the magnitude is the angle
#             turned anticlockwise in radians around the vector's
#             direction.
#     """
#     return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)



def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))



def convert_euler_to_6D(eulers, n_joints):
    # euler: (n_frames, 75*3,)
    # return: (n_frames, 75*6, )
    # breakpoint()

    # Assumption that the order of euler angles is XYZ
    # convert euler angle to rotation matrix
    eulers = eulers.reshape(-1, n_joints, 3) # (n_frames, 75, 3)
    eulers = (eulers / 180) * np.pi # (n_frames, 75, 3)
    eulers = torch.from_numpy(eulers).float()

    rot_mats = euler_angles_to_matrix(eulers, 'XYZ') # (n_frames, 75, 3, 3)
    rep_6d = matrix_to_rotation_6d(rot_mats) # (n_frames, 75, 6)
    rep_6d = rep_6d.reshape(-1, n_joints*6) # (n_frames, 75*6)
    rep_6d = rep_6d.numpy()
    return rep_6d


def convert_6D_to_euler(rep_6d, n_joints):
    # rep_6d: (n_frames, 75*6,)
    # return: (n_frames, 75*3, )
    # breakpoint()

    # Assumption that the order of euler angles is XYZ
    # convert 6D representation to rotation matrix
    rep_6d = rep_6d.reshape(-1, n_joints, 6) # (n_frames, 75, 6)
    rep_6d = torch.from_numpy(rep_6d).float()

    rot_mats = rotation_6d_to_matrix(rep_6d) # (n_frames, 75, 3, 3)
    eulers = matrix_to_euler_angles(rot_mats, 'XYZ') # (n_frames, 75, 3)
    eulers = (eulers * 180) / np.pi # (n_frames, 75, 3) # convert to degrees

    eulers = eulers.reshape(-1, n_joints*3) # (n_frames, 75*3)
    eulers = eulers.numpy()
    return eulers


def forward_kinematics_cont6d(cont6d_params, root_pos, offset, kinematic_tree, skel_joints=None, do_root_R=True):
    # cont6d_params (batch_size, joints_num, 6)
    # joints (batch_size, joints_num, 3)
    # root_pos (batch_size, 3)
    
    offsets = offset.expand(cont6d_params.shape[0], -1, -1)
    joints = torch.zeros(cont6d_params.shape[:-1] + (3,)).to(cont6d_params.device)
    joints[..., 0, :] = root_pos
    for chain in kinematic_tree:
        if do_root_R:
            matR = rotation_6d_to_matrix(cont6d_params[:, 0])
        else:
            matR = torch.eye(3).expand((len(cont6d_params), -1, -1)).detach().to(cont6d_params.device)
        for i in range(1, len(chain)):
            matR = torch.matmul(rotation_6d_to_matrix(cont6d_params[:, chain[i]]), matR)
            offset_vec = offsets[:, chain[i]].unsqueeze(-1)
            # print(matR.shape, offset_vec.shape)
            joints[:, chain[i]] = torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
    return joints

def forward_kinematics_euler(eulers, root_pos, offset, kinematic_tree, skel_joints=None, do_root_R=True):
    # eulers (batch_size, joints_num, 3)
    # joints (batch_size, joints_num, 3)
    # root_pos (batch_size, 3)
    offsets = offset.expand(eulers.shape[0], -1, -1)
    joints = torch.zeros(eulers.shape[:-1] + (3,)).to(eulers.device)
    joints[..., 0, :] = root_pos
    for chain in kinematic_tree:
        if do_root_R:
            matR = euler_angles_to_matrix(eulers[:, 0], 'XYZ')
        else:
            matR = torch.eye(3).expand((len(eulers), -1, -1)).detach().to(eulers.device)
        for i in range(1, len(chain)):
            matR = torch.matmul(matR, euler_angles_to_matrix(eulers[:, chain[i]], 'XYZ'))
            offset_vec = offsets[:, chain[i]].unsqueeze(-1)
            # print(matR.shape, offset_vec.shape)
            joints[:, chain[i]] = torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
    return joints