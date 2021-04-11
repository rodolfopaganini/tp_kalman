# TP Kalman filter
# Author: Rodolfo Paganini

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def main():
    # Load position data
    original_position = np.loadtxt('posicion.dat')

    # Convert the data into separate arrays per coordinate
    t = original_position[np.arange(len(original_position)), [0]]
    x = original_position[np.arange(len(original_position)), [1]]
    y = original_position[np.arange(len(original_position)), [2]]
    z = original_position[np.arange(len(original_position)), [3]]

    # Define 'measurement data', which is the real one plus noise
    x_measurements = x + np.random.normal(scale=10, size=t.size)
    y_measurements = y + np.random.normal(scale=10, size=t.size)
    z_measurements = z + np.random.normal(scale=10, size=t.size)

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
    output_matrix = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
    ])
    process_noise_cov = 0.3 * np.identity(9)
    observation_noise_cov = 100 * np.identity(3)
    input_values = np.zeros(9)

    # Set the initial state
    estimated_state = np.array([10.7533, 36.6777, -45.1769, 1.1009, -17.0, 35.7418, -5.7247, 3.4268, 5.2774])
    error_cov = np.identity(9)
    np.fill_diagonal(error_cov, [100, 100, 100, 1, 1, 1, 0.01, 0.01, 0.01])

    # Predict values using a kalman filter
    x_predicted = []
    y_predicted = []
    z_predicted = []
    for x_measurement, y_measurement, z_measurement in zip(x_measurements, y_measurements, z_measurements):
        measured_output = np.array([x_measurement, y_measurement, z_measurement])
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
        x_predicted.append(estimated_state[0])
        y_predicted.append(estimated_state[1])
        z_predicted.append(estimated_state[2])

    # Plot the original values, the noisy ones and the predicted ones
    for variable_name in ['x', 'y', 'z']:
        # Set the figure up
        fig, ax = plt.subplots()
        ax.plot(t, eval(variable_name), label='original')
        ax.plot(t, eval(variable_name + '_measurements'), label='measurements')
        ax.plot(t, eval(variable_name + '_predicted'), label='predicted')
        ax.set(
            xlabel='time (s)',
            ylabel='{} (m)'.format(variable_name),
            title='Position in {} axis'.format(variable_name),
        )
        ax.grid()
        plt.legend()
        fig.savefig("{}.png".format(variable_name))
        plt.show()


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
    error_cov_posteriori = np.matmul(np.identity(9) - np.matmul(kalman_gain, output_matrix), error_cov_priori)  # TODO make this dependent on the state matrix dims
    return estimated_state_posteriori, error_cov_posteriori


if __name__ == "__main__":
    main()
