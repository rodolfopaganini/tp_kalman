# TP Kalman filter
# Author: Rodolfo Paganini

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def main():
    # Load position and velocity data
    original_position = np.loadtxt('posicion.dat')
    original_velocity = np.loadtxt('velocidad.dat')

    # Convert position data into separate arrays per coordinate
    t = original_position[np.arange(len(original_position)), [0]]
    x = original_position[np.arange(len(original_position)), [1]]
    y = original_position[np.arange(len(original_position)), [2]]
    z = original_position[np.arange(len(original_position)), [3]]

    # Convert velocity data into separate arrays per coordinate
    vx = original_velocity[np.arange(len(original_position)), [1]]
    vy = original_velocity[np.arange(len(original_position)), [2]]
    vz = original_velocity[np.arange(len(original_position)), [3]]

    # Define state and output matrices
    state_matrix = np.array(
        [
            [1, 0, 0, 1, 0, 0, 0.5, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0.5, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0.5],
            [0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
    )
    input_matrix = np.identity(9)
    process_noise_cov = 0.3 * np.identity(9)
    input_values = np.zeros(9)

    uniform_limits = 10 * math.sqrt(3)
    for description, measured_variables_names, measurements, output_matrix, observation_noise_cov in [
        [
            'with gaussian noise',
            ['x', 'y', 'z'],
            (
                x + np.random.normal(scale=10, size=t.size),
                y + np.random.normal(scale=10, size=t.size),
                z + np.random.normal(scale=10, size=t.size),
            ),
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
            ]),
            100 * np.identity(3),
        ],
        [
            'with uniform noise',
            ['x', 'y', 'z'],
            (
                x + np.random.uniform(high=uniform_limits, low=-uniform_limits, size=t.size),
                y + np.random.uniform(high=uniform_limits, low=-uniform_limits, size=t.size),
                z + np.random.uniform(high=uniform_limits, low=-uniform_limits, size=t.size),
            ),
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
            ]),
            100 * np.identity(3),
        ],
        [
            'with gaussian noise and speed measurement',
            ['x', 'y', 'z', 'vx', 'vy', 'vz'],
            (
                x + np.random.normal(scale=10, size=t.size),
                y + np.random.normal(scale=10, size=t.size),
                z + np.random.normal(scale=10, size=t.size),
                vx + np.random.normal(scale=0.2, size=t.size),
                vy + np.random.normal(scale=0.2, size=t.size),
                vz + np.random.normal(scale=0.2, size=t.size),
            ),
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]),
            np.array([
                [100, 0, 0, 0, 0, 0],
                [0, 100, 0, 0, 0, 0],
                [0, 0, 100, 0, 0, 0],
                [0, 0, 0, 0.04, 0, 0],
                [0, 0, 0, 0, 0.04, 0],
                [0, 0, 0, 0, 0, 0.04],
            ]),
        ],
    ]:
        # Set the initial state
        estimated_state = np.array([10.7533, 36.6777, -45.1769, 1.1009, -17.0, 35.7418, -5.7247, 3.4268, 5.2774])
        error_cov = np.identity(9)
        np.fill_diagonal(error_cov, [100, 100, 100, 1, 1, 1, 0.01, 0.01, 0.01])

        # Predict values using a kalman filter
        predicted = [list() for i in range(len(measured_variables_names))]
        for measurement in zip(*measurements):
            measured_output = np.array(measurement)
            estimated_state, error_cov = kalman(
                state_matrix=state_matrix,
                input_matrix=input_matrix,
                output_matrix=output_matrix,
                measured_output=measured_output,
                input_values=input_values,
                prev_estimated_state=estimated_state,
                prev_error_cov=error_cov,
                process_noise_cov=process_noise_cov,
                observation_noise_cov=observation_noise_cov,
            )
            for variable in range(len(measured_variables_names)):
                predicted[variable].append(estimated_state[variable])

        # Generate the figures
        for i, variable_name in enumerate(measured_variables_names):
            # Plot the original values, the noisy ones and the predicted ones
            fig, ax = plt.subplots()
            ax.plot(t, eval(variable_name), label='original')
            ax.plot(t, measurements[i], label='measurements')
            ax.plot(t, predicted[i], label='predicted')
            ax.set(
                xlabel='time (s)',
                ylabel='{} (m)'.format(variable_name),
                title='{} in {} axis'.format('Position' if i < 3 else 'Velocity', variable_name[-1]),
            )
            ax.grid()
            plt.legend()
            fig.savefig("{}_{}.svg".format(description, variable_name))
            # plt.show()

            # Plot the errors in the 3 coordinates
            fig, ax = plt.subplots()
            error = eval(variable_name) - predicted[i]
            ax.plot(t, error, label=variable_name)
            ax.set(
                xlabel='time (s)',
                ylabel='Error (m)',
                title='Error in {} axis'.format(variable_name[-1]),
            )
            ax.grid()
            plt.legend()
            fig.savefig("error_{}_{}.svg".format(description, variable_name))
            # plt.show()
            print('Error {} {}: Mean: {}, RMS: {}'.format(
                description,
                variable_name,
                np.mean(error),
                np.sqrt(sum(error ** 2) / len(error)),
            ))


def kalman(
        state_matrix,
        input_matrix,
        output_matrix,
        measured_output,
        input_values,
        prev_estimated_state,
        prev_error_cov,
        process_noise_cov,
        observation_noise_cov,
):
    """
    This function implements a kalman filter
    :param state_matrix: state matrix
    :param input_matrix: input matrix
    :param output_matrix: output matrix
    :param measured_output: measured output
    :param input_values: manipulated variables
    :param prev_estimated_state: estimated state at the previous time step
    :param prev_error_cov: error covariance in the previous time step
    :param process_noise_cov: process noise covariance
    :param observation_noise_cov: observation noise covariance
    :return:
        - estimated_state_posteriori: the estimate of the states at the current time step
        - error_cov_posteriori: error covariance at the current time step
    """

    # A priori step
    estimated_state_priori = np.matmul(state_matrix, prev_estimated_state) + np.matmul(input_matrix, input_values)
    error_cov_priori = np.matmul(
        state_matrix,
        np.matmul(prev_error_cov, state_matrix.T)) \
        + np.matmul(input_matrix, np.matmul(process_noise_cov, input_matrix.T))

    # Correction step
    kalman_gain = np.matmul(
        error_cov_priori,
        np.matmul(
            output_matrix.T,
            np.linalg.inv(
                observation_noise_cov
                + np.matmul(output_matrix, np.matmul(error_cov_priori, output_matrix.T)),
            ),
        ),
    )
    estimated_state_posteriori = estimated_state_priori + np.matmul(
        kalman_gain,
        measured_output - np.matmul(output_matrix, estimated_state_priori),
    )
    error_cov_posteriori = np.matmul(
        np.identity(len(state_matrix)) - np.matmul(kalman_gain, output_matrix),
        error_cov_priori,
    )
    return estimated_state_posteriori, error_cov_posteriori


if __name__ == "__main__":
    main()
