from qiskit import QuantumCircuit, execute, Aer, transpile

from qiskit.result.result import Result
from qiskit.providers.aer.noise import NoiseModel

from qiskit.transpiler import PassManager, PassManagerConfig, CouplingMap
from qiskit.transpiler.passes import Unroller, Optimize1qGates
from qiskit.transpiler.preset_passmanagers.level3 import level_3_pass_manager

from numpy import asarray, ndarray, shape, zeros, empty, average, transpose, dot
from numpy.linalg import solve

import random, os, pickle, sys, errno

abs_path = os.path.dirname(__file__)
sys.path.append(abs_path)
sys.path.append(os.path.dirname(abs_path))

from zero_noise_extrapolation import Richardson_extrapolate

from typing import Callable, Union

"""
-- ZERO NOISE EXTRAPOLATION for CNOT-gates --

This is an implementation of the zero-noise extrapolation technique for quantum error mitigation. The goal is to
mitigate noise present in a quantum device when evaluating some expectation value that is computed by a quantum circuit
with subsequent measurements. The main idea of zero-noise extrapolation is to amplify the noise by a set of known
noise amplification factors, such as to obtain a set of noise amplified expectation values. Richardsson extrapolation
is then used to extrapolate the expectation value to the zero-noise limit.

The noise that is here amplified and mitigated is specifically general noise in CNOT-gates. Note that in modern quantum
devices the noise in the multi-qubit CNOT-gates tend to be an order of magnitude larger than in single-qubit gates.

To amplify the noise we use that the CNOT-gate is its own inverse, i.e., CNOT*CNOT = Id, where Id is the identity gate.
Thus an odd number of CNOT-gates in a series will in the noise-less case have the same action as a single CNOT-gate.
The noise is amplified by replacing each CNOT-gate in the original bare circuit with a series of (2*i + 1) CNOT's,
using noise amplification factors c=1, 3, 5, ..., 2*n - 1, for n being the total number of amplification factors.
For c=3, each CNOT is thus replaced by the sequence CNOT*CNOT*CNOT, and while this has the same action in the noise-less
case, in the noisy case the noise operation associated with the noisy CNOT-gate will be applied thrice instead of once.

"""


class ZeroNoiseExtrapolation:

    def __init__(self, qc: QuantumCircuit, exp_val_func: Callable, backend=None, exp_val_option: dict = None,
                 noise_model: Union[NoiseModel, dict] = None, n_amp_factors: int = 3, shots: int = 8192,
                 pauli_twirl: bool = False, pass_manager: PassManager = None,
                 save_results: bool = False, experiment_name: str = "", option: dict = None):
        """
        CONSTRUCTOR

        Parameters
        ----------
        qc : qiskit.QuantumCircuit
            A complete quantum circuit, with measurements, that we want to perform quantum error mitigation on.
            When run on a quantum backend, the circuit should output a set of measurements results from which the
            desired expectation value can be estimated.

        exp_val_func : Callable
            A function that computes the desired expectation value based on the measurement results outputted by the
            quantum circuit. The function should take two arguments: a qiskit.result.result.Experiment object as its
            first argument, and a dictionary with possible options as its second.

        backend : A valid qiskit backend, IBMQ device or simulator
            A qiskit backend, either an IBMQ quantum backend or a simulator backend, for circuit executions.
            If none is passed, the qasm_simulator will be used.

        exp_val_option : dict, (optional)
            Options for the exp_val_func expectation value function

        noise_model : qiskit.providers.aer.noise.NoiseModel or dict, (optional)
            Custom noise model for circuit executions with the qasm simulator backend.

        n_amp_factors : int, (default set to 3)
            The number of noise amplification factors to be used. For n number of amplification factors, the specific
            noise amplification factors will be [1, 3, 5, ..., 2*n - 1]. Larger amounts of noise amplification factors
            tend to give better results, but slower convergence thus requiring large amounts of shots.
            Higher noise amplification also increases circuit depth, scaling linearly with the amplification factor c_i,
            and at some point the circuit depth and the consecutive decoherence will eliminate any further advantage.

        shots: int, (default set to 8192)
            The number of "shots" of each experiment to be executed, where one experiment is a single execution of a
            quantum circuit. To obtain an error mitigated expectation value, a total of shots*n_amp_factors experiments
            is performed.

        pauli_twirl : bool, (optional)
            Perform Pauli twirling of each noise amplified circuit, True / False.

        pass_manager: qiskit.transpiler.PassManager, (optional)
            Optional custom pass_manager for circuit transpiling. If none is passed, the circuit will be transpiled
            using the qiskit optimization_level=3 preset, which is the heaviest optimization preset.

        save_results: bool, (optional)
            If True, will attempt to read transpiled circuit and experiment results for each noise amplified from disk,
            and if this fails, the transpiled circuit and/or experiment measurement results will be saved to disk.

        experiment_name: string, (optional)
            The experiment name used when reading and writing transpiled circuits and measurement results to disk.
            The experiment name will form the base for the full filename for each written/read file.
            This argument is required if save_result = True.

        option: dict, (optional)
            Options for the writing/reading of transpiled circuits and measurement results.
            option["directory"] gives the directory in which files will be written to/attempted to be read from.
            If no option is passed, the default directory used will be option["directory"] = "results".

        """

        # Set backend for circuit execution. If none is passed, use the qasm_simulator backend.
        if backend is None:
            self.backend = Aer.get_backend("qasm_simulator")
            self.is_simulator = True
        else:
            self.backend = backend
            self.is_simulator = backend.configuration().simulator

        self.exp_val_func = exp_val_func
        self.exp_val_option = exp_val_option

        self.noise_model = noise_model

        self.n_amp_factors = n_amp_factors
        self.noise_amplification_factors = asarray([(2*i + 1) for i in range(0, n_amp_factors)])

        self.pauli_twirl = pauli_twirl

        # Max number of shots for one circuit execution on IBMQ devices is 8192.
        # To do more shots, we have to partition the total experiment into several executions.
        self.shots, self.repeats = self.partition_shots(shots)

        # Variables involved in writing and reading results to/from disk
        self.save_results, self.option = save_results, option
        if self.option is None:
            self.option = {}
        self.experiment_name = ""
        if self.save_results:
            self.set_experiment_name(experiment_name)
            self.create_directory()

        # Initial transpiling of the quantum circuit. If no custom pass manager is passed, the optimization_level=3
        # qiskit preset (the heaviest optimization preset) will be used. If save_results=True, will attempt to read
        # the transpiled circuit from disk.
        circuit_read_from_file = False
        if self.save_results:
            qc_from_file = self.read_from_file(self.experiment_name + ".circuit")
            if not (qc_from_file is None):
                circuit_read_from_file = True
                self.qc = qc_from_file
        if not circuit_read_from_file:
            self.qc = self.transpile_circuit(qc, custom_pass_manager=pass_manager)

        """
            Initialization of other variables for later use:
        """

        self.counts = []

        self.depths, self.gamma_factors = empty(n_amp_factors), empty(n_amp_factors)

        # Will store expectation values for each individual circuit execution
        self.all_exp_vals = zeros(0)
        # Will store expectation values for each noise amplified circuit, averaged over all sub-executions
        self.noise_amplified_exp_vals = zeros(0)

        self.result = None

        self.measurement_results = []

    def partition_shots(self, tot_shots: int) -> (int, int):
        """
        IBMQ devices limits circuit executions to a max of 8192 shots per experiment. To perform more than 8192 shots,
        the experiment has to be repeated. Therefore, if shots > 8192, we partition the execution into several repeats
        of less than 8192 shots each.

        Parameters
        ----------
        tot_shots : int
            The total number of circuit execution shots.

        Returns
        -------
        shots, repeats: (int, int)
            Shots per repeat, number of repeats
        """
        if tot_shots <= 8192:
            return tot_shots, 1
        else:
            repeats = (tot_shots // 8192) + 1
            return int(tot_shots / repeats), repeats

    def set_shots(self, shots: int):
        self.shots, self.repeats = self.partition_shots(shots)

    def get_shots(self):
        return self.repeats * self.shots

    def set_experiment_name(self, experiment_name):
        """
        Construct the experiment name that will form the base for the filenames that will be read from / written to
        when save_results=True. The full experiment name will contain information about the backend, number of shots,
        and pauli twirling, to ensure that different experiments using different parameters don't read from the same
        data.

        Parameters
        ----------
        experiment_name : str
            The base for the experiment name that will be used for filenames

        """
        if self.save_results and experiment_name == "":
            raise Exception("experiment_name cannot be empty when writing/reading results from disk is activated")
        self.experiment_name = experiment_name
        self.experiment_name += "__ZNE_CNOT_REP_"
        self.experiment_name += "_backend" + self.backend.name()
        self.experiment_name += "_shots" + str(self.get_shots())
        self.experiment_name += "_paulitwirling" + str(self.pauli_twirl)

    def create_directory(self):
        """
        Attempt to create the directory in which to read from/write to files. The case whereby the directory already
        exists is handled by expection handling.

        The directory is given by self.option["directory"], with the default being "results".

        """
        if not self.save_results:
            return
        directory = self.option.get("directory", "results")
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e

    def read_from_file(self, filename: str):
        """
        Attempts to read data for a given filename, looking in the directory given by self.option["directory"].

        Parameters
        ----------
        filename : str
            The full filename for the file in question

        Returns
        -------
        data : any
            The data read from said file. None if the file wasn't found
        """
        directory = self.option.get("directory", "results")
        if os.path.isfile(directory + "/" + filename):
            file = open(directory + "/" + filename, "rb")
            data = pickle.load(file)
            file.close()
            return data
        else:
            return None

    def write_to_file(self, filename: str, data):
        """
        Writes data to file with given filename, located in the directory given by self.option["directory"].

        Parameters
        ----------
        filename : str
            The full filename of the file to be written to.
        data : any
            The data to be stored.

        """
        directory = self.option.get("directory", "results")
        file = open(directory + "/" + filename, "wb")
        pickle.dump(data, file)
        file.close()

    def noise_amplify_and_pauli_twirl_cnots(self, qc: QuantumCircuit, amp_factor: int,
                                            pauli_twirl: bool) -> QuantumCircuit:
        """
        Amplify CNOT-noise by extending each CNOT-gate as CNOT^amp_factor and possibly Pauli-twirl all CNOT-gates

        Using CNOT*CNOT = I, the identity, and an amp_factor = (2*n + 1) for an integer n, then the
        extended CNOT will have the same action as a single CNOT, but with the noise amplified by
        a factor amp_factor.

        :param qc: Quantum circuit for which to Pauli twirl all CNOT gates and amplify CNOT-noise
        :param amp_factor: The noise amplification factor, must be (2n + 1) for n = 0,1,2,3,...
        :param pauli_twirl: Add pauli twirling True / False
        :return: Noise-amplified and possibly Pauli-twirled Quantum Circuit
        """

        if (amp_factor - 1) % 2 != 0:
            raise Exception("Invalid amplification factors", amp_factor)

        # The circuit may be expressed in terms of various types of gates.
        # The 'Unroller' transpiler pass 'unrolls' (decomposes) the circuit gates to be expressed in terms of the
        # physical gate set [u1,u2,u3,cx]

        # The cz, cy (controlled-Z and -Y) gates can be constructed from a single cx-gate and sinlge-qubit gates.
        # For backends with native gate sets consisting of some set of single-qubit gates and either the cx, cz or cy,
        # unrolling the circuit to the ["u3", "cx"] basis, amplifying the cx-gates, then unrolling back to the native
        # gate set and doing a single-qubit optimization transpiler pass, is thus still general.

        unroller_ugatesandcx = Unroller(["u1", "u2", "u3", "cx"])
        pm = PassManager(unroller_ugatesandcx)

        unrolled_qc = pm.run(qc)

        circuit_qasm = unrolled_qc.qasm()
        new_circuit_qasm_str = ""

        qreg_name = find_qreg_name(circuit_qasm)

        for i, line in enumerate(circuit_qasm.splitlines()):
            if line[0:2] == "cx":
                for j in range(amp_factor):
                    if pauli_twirl:
                        new_circuit_qasm_str += pauli_twirl_cnot_gate(qreg_name, line)
                    else:
                        new_circuit_qasm_str += (line + "\n")
            else:
                new_circuit_qasm_str += line + "\n"

        new_qc = QuantumCircuit.from_qasm_str(new_circuit_qasm_str)

        # The "Optimize1qGates" transpiler pass optimizes adjacent single-qubit gates, for a native gate set with the
        # u3 gates it collapses any chain of adjacent single-qubit gates into a single, equivalent u3-gate.
        # We want to collapse unnecessary single-qubit gates to minimize circuit depth, but not CNOT-gates
        # as these give us the noise amplification.
        unroller_backendspecific = Unroller(self.backend.configuration().basis_gates)
        optimize1qates = Optimize1qGates()

        pm = PassManager([unroller_backendspecific, optimize1qates])

        return pm.run(new_qc)

    def transpile_circuit(self, qc: QuantumCircuit, custom_pass_manager: PassManager = None) -> QuantumCircuit:
        """
        Transpile and optimize the input circuit, optionally by  using a custom pass manager.
        If no custom pass manager is given, the optimization_level = 3 preset for the qiskit transpiler,
        the heaviest optimization preset, will be used.

        As we want to add additional CNOTs for noise amplification and possibly additional single qubit gates
        for Pauli twirling, we need to transpile the circuit before both the noise amplification is applied and
        before circuit execution. This is to avoid the additional CNOT-gates beinng removed by the transpiler.

        The Optimize1qGates transpiler pass will be used later to optimize single qubit gates added during
        the Pauli-twirling, as well as the Unroller pass which merely decomposes the given circuit gates into
        the given set of basis gates.

        Parameters
        ----------
        qc : qiskit.QuantumCircuit
            The original bare quantum circuit.
        custom_pass_manager : qiskit.transpiler.PassManager, (optional)
            A custom pass manager to be used in transpiling.

        Returns
        -------
        transpiled_circuit : qiskit.QuantumCircuit
            The transpiled quantum circuit.
        """

        if custom_pass_manager == None:
            pass_manager_config = PassManagerConfig(basis_gates=["id", "u1", "u2", "u3", "cx"],
                                                    backend_properties=self.backend.properties())
            if not self.is_simulator:
                pass_manager_config.coupling_map = CouplingMap(self.backend.configuration().coupling_map)

            pass_manager = level_3_pass_manager(pass_manager_config)
        else:
            pass_manager = custom_pass_manager

        self.passes = pass_manager.passes()  # Saves the list of passes used for transpiling

        transpiled_circuit = pass_manager.run(qc)

        # As there is some randomness involved in the qiskit transpiling we might want to save
        # the specific transpiled circuit that is used in order to access it later.
        if self.save_results:
            filename = self.experiment_name + ".circuit"
            self.write_to_file(filename, transpiled_circuit)

        return transpiled_circuit

    def execute_circuit(self, qc: QuantumCircuit, shots=None) -> Result:
        """

        Parameters
        ----------
        qc : qiskit.QuantumCircuit
            The specific quantum circuit to be executed.
        shots : int, (optional)
            The number of shots of the circuit execution. If none is passed, self.shots is used.
        Returns
        -------
        circuit_measurement_results : qiskit.result.result.Result
            A Result object containing the data and measurement results for the circuit executions.
        """

        if shots is None:
            shots = self.shots
            repeats = self.repeats
        else:
            shots, repeats = self.partition_shots(shots)

        # The max number of shots on a single execution on the IBMQ devices is 8192.
        # If shots > 8192, we have to partition the execution into several sub-executions.
        # Note that several circuits can be entered into the IBMQ queue at once by passing them in a list.
        execution_circuits = [qc.copy() for i in range(repeats)]

        # non-simulator backends throws unexpected argument when passing noise_model argument to them
        if self.is_simulator:
            job = execute(execution_circuits, backend=self.backend, noise_model=self.noise_model,
                          pass_manager=PassManager(), shots=shots)
        else:
            job = execute(execution_circuits, backend=self.backend,
                          pass_manager=PassManager(), shots=shots)

        circuit_measurement_results = job.result()

        return circuit_measurement_results

    def compute_exp_val(self, result: Result) -> (float, ndarray):
        """

        Parameters
        ----------
        result : qiskit.result.result.Result
            A qiskit Result object containing all measurement results from a set of quantum circuit executions.
        Returns
        -------
        averaged_experiment_exp_vals, experiment_exp_vals : Tuple[float, ndarray]
            The final experiment expectation value, averaged over all circuit sub-executions, and a numpy array
            containing the expectation values for each circuit sub-execution.
        """
        experiment_exp_vals = zeros(len(result.results))
        for i, experiment_result in enumerate(result.results):
            experiment_exp_vals[i] = self.exp_val_func(experiment_result, self.exp_val_option)
        return average(experiment_exp_vals), experiment_exp_vals

    def mitigate(self, verbose: bool = False) -> float:
        """
        Perform the quantum error mitigation.

        Parameters
        ----------
        verbose : bool
            Do prints throughout the computation.

        Returns
        -------
        result : float
            The mitigated expectation value.
        """

        n_amp_factors = shape(self.noise_amplification_factors)[0]

        self.noise_amplified_exp_vals = zeros((n_amp_factors,))
        self.all_exp_vals = zeros((n_amp_factors, self.repeats))

        if verbose:
            print("Shots per circuit=", self.repeats * self.shots, ", executed as ", self.shots, " per repeat for ",
                  self.repeats, " experiment repeats. Pauli twirl=", self.pauli_twirl,
                  "\nNumber of noise amplification factors=", n_amp_factors,
                  "\nNoise amplification factors=", self.noise_amplification_factors, sep="")

        if verbose:
            print("-----\nConstructing circuits")

        noise_amplified_circuits = []

        for j, amp_factor in enumerate(self.noise_amplification_factors):
            noise_amplified_circuits.append(self.noise_amplify_and_pauli_twirl_cnots(qc=self.qc, amp_factor=amp_factor,
                                                                                     pauli_twirl=self.pauli_twirl))
            self.depths[j] = noise_amplified_circuits[-1].depth()

        if verbose:
            print("Depths=", self.depths, sep="")

            print("-----\nExecuting circuits")

        for i in range(n_amp_factors):
            print("Noise amplification factor ", i + 1, " of ", n_amp_factors)

            circuit_measurement_results, circuit_read_from_file = None, False

            if self.save_results:
                tmp = self.read_from_file(self.experiment_name + "_r{:}.results".format(self.noise_amplification_factors[i]))
                if tmp != None:
                    circuit_measurement_results = tmp
                    circuit_read_from_file = True
                    if verbose:
                        print("Results successfully read")
                else:
                    if verbose:
                        print("Results not found")

            # circuit_read_from_file equals False if either the option for reading from/writing to disk is off, e.g.,
            # if save_results=Flase, or if the result was attempted to be read but not found.
            # In either case the experiment must be run from scratch.
            if not circuit_read_from_file:
                circuit_measurement_results = self.execute_circuit(noise_amplified_circuits[i])
                if self.save_results:
                    # Write the noise amplified expectation value to disk
                    self.write_to_file(self.experiment_name + "_r{:}.results".format(self.noise_amplification_factors[i]),
                                       circuit_measurement_results)

            self.noise_amplified_exp_vals[i], self.all_exp_vals[i, :] = self.compute_exp_val(
                circuit_measurement_results)

            self.measurement_results.append(circuit_measurement_results)

        # Find the mitigated expectation value by Richardson extrapolation
        result,_ = Richardson_extrapolate(self.noise_amplified_exp_vals, self.noise_amplification_factors)
        self.result = result[0]

        if verbose:
            print("-----", "\nError mitigation done",
                  "\nBare circuit expectation value: ", self.noise_amplified_exp_vals[0],
                  "\nNoise amplified expectation values: ",self.noise_amplified_exp_vals,
                  "\nMitigated expectation value:", self.result, "\n-----", sep="")

        return self.result


# PAULI TWIRLING AND NOISE AMPLIFICATION HELP FUNCTIONS

# Conversion from pauli x/y/z-gates to physical u1/u3-gates in correct OpenQASM-format
PHYSICAL_GATE_CONVERSION = {"X": "u3(pi,0,pi)", "Z": "u1(pi)", "Y": "u3(pi,pi/2,pi/2)"}


def find_qreg_name(circuit_qasm: str) -> str:
    """
    Finds the name of the quantum register in the circuit. Assumes a single quantum register.

    Parameters
    ----------
    circuit_qasm : str
        The OpenQASM-string for the circuit

    Returns
    -------
    qreg_name :str
        The name of the quantum register
    """
    for line in circuit_qasm.splitlines():
        if line[0:5] == "qreg ":
            qreg_name = ""
            for i in range(5, len(line)):
                if line[i] == "[" or line[i] == ";":
                    break
                elif line[i] != " ":
                    qreg_name += line[i]
            return qreg_name


def find_cnot_control_and_target(qasm_line: str) -> (int, int):
    """
    Find the indices of the control and target qubits for a specific CNOT-gate.

    Parameters
    ----------
    qasm_line : str
        The line containing the CNOT-gate in question taken from the OpenQASM-format string of the quantum circuit.

    Returns
    -------
    control, target : Tuple[int, int]
        qubit indices for control and target qubits
    """
    qubits = []
    for i, c in enumerate(qasm_line):
        if c == "[":
            qubit_nr = ""
            for j in range(i + 1, len(qasm_line)):
                if qasm_line[j] == "]":
                    break
                qubit_nr += qasm_line[j]
            qubits.append(int(qubit_nr))
    return qubits[0], qubits[1]


def propagate(control_in: str, target_in: str):
    """
    Finds the c,d gates such that (a x b) CNOT (c x d) = CNOT for an ideal CNOT-gate, based on the a (control_in)
    and b (target_in) pauli gates by "propagating" the a,b gates over a CNOT-gate by the following identities:

    (X x I) CNOT = CNOT (X x X)
    (I x X) CNOT = CNOT (I x X)
    (Z x I) CNOT = CNOT (I x Z)
    (I x Z) CNOT = XNOT (Z x Z)

    Note that instead of Pauli-twirling with [X,Z,Y] we use [X,Z,XZ] where XZ = -i*Y.
    The inverse of XZ is ZX = -XZ = i*Y. The factors of plus minus i are global phase factors which can be ignored.

    Parameters
    ----------
    control_in : str
        The Pauli operator on control qubit before the CNOT, i.e., a
    target_in : str
        The Pauli operator on target qubit before the CNOT, i.e., b
    Returns
    -------
    control_out, target_out : Tuple[str, str]
        The operators c and d such that (a x b) CNOT (c x d) = CNOT
    """

    control_out, target_out = '', ''
    if 'X' in control_in:
        control_out += 'X'
        target_out += 'X'
    if 'X' in target_in:
        target_out += 'X'
    if 'Z' in control_in:
        control_out += 'Z'
    if 'Z' in target_in:
        control_out += 'Z'
        target_out += 'Z'

    # Pauli gates square to the identity, i.e. XX = I, ZZ = I
    # Remove all such occurences from the control & target out Pauli gate strings
    if 'ZZ' in control_out:
        control_out = control_out[:-2]
    if 'ZZ' in target_out:
        target_out = target_out[:-2]
    if 'XX' in control_out:
        control_out = control_out[2:]
    if 'XX' in target_out:
        target_out = target_out[2:]

    # If no Pauli gates remain then we have the identity gate I
    if control_out == '':
        control_out = 'I'
    if target_out == '':
        target_out = 'I'

    # The inverse of XZ is ZX, therefore we reverse the gate order to obtain the correct pauli gates c,d
    # such that (a x b) CNOT (c x d) = CNOT (c^-1 x d^-1) (c x d) = CNOT
    return control_out[::-1], target_out[::-1]


def apply_qasm_pauli_gate(qreg_name: str, qubit: int, pauli_gates: str):
    """
    Construct an OpenQASM-string line with the given Pauli-gates applid to the given qubit.

    Parameters
    ----------
    qreg_name :str
        The name of the qiskit.QuantumRegister containing the qubit.
    qubit : int
        The index of the qubit.
    pauli_gates : str
        A string determining the Pauli-gates to be applied. Must be a sequence the characters I, X, Y and/or Z.
    Returns
    -------
    new_qasm_line : str
        An OpenQASM string with the Pauli-gates applied to the qubit.
    """
    new_qasm_line = ''
    for gate in pauli_gates:
        if gate != 'I':
            if gate not in PHYSICAL_GATE_CONVERSION.keys():
                raise Exception("Invalid Pauli-gate used in Pauli-twirl: {:}".format(gate))
            u_gate = PHYSICAL_GATE_CONVERSION[gate]
            new_qasm_line += u_gate + ' ' + qreg_name + '[' + str(qubit) + '];' + '\n'
    return new_qasm_line


def pauli_twirl_cnot_gate(qreg_name: str, qasm_line_cnot: str) -> str:
    """
    Pauli-twirl a specific CNOT-gate. This involves drawing two random Pauli-gates a and b, picked from the single-qubit
    Pauli set {Id, X, Y, Z}, then determining the corresponding two Pauli-gates c and d such that
    (a x b) * CNOT * (c x d) = CNOT, for an ideal CNOT.

    The original CNOT gates is then replaced by the gate ((a x b) * CNOT * (c x d)). This transforms the noise in the
    CNOT-gate into stochastic Pauli-type noise. An underlying assumption is that the noise in the single-qubit Pauli
    gates is negligible to the noise in the CNOT-gates.

    Parameters
    ----------
    qreg_name : str
        The name of the qiskit.QuantumRegister for the qubits in question.
    qasm_line_cnot : str
        The OpenQASM-string line containing the CNOT-gate.

    Returns
    -------
    new_qasm_line : str
        A new OpenQASM-string section to replace the aforementioned OpenQASM line containing the CNOT-gate, where not
        the CNOT-gate has been Pauli-twirled.
    """

    control, target = find_cnot_control_and_target(qasm_line_cnot)

    # Note: XZ = -i*Y, with inverse (XZ)^-1 = ZX = i*Y. This simplifies the propagation of gates a,b over the CNOT
    pauli_gates = ["I", "X", "Z", "XZ"]

    a = random.choice(pauli_gates)
    b = random.choice(pauli_gates)

    # Find gates such that:
    # (a x b) CNOT (c x d) = CNOT for an ideal CNOT-gate,
    # by propagating the Pauli gates through the CNOT

    c, d = propagate(a, b)

    new_qasm_line = apply_qasm_pauli_gate(qreg_name, control, a)
    new_qasm_line += apply_qasm_pauli_gate(qreg_name, target, b)
    new_qasm_line += qasm_line_cnot + '\n'
    new_qasm_line += apply_qasm_pauli_gate(qreg_name, target, d)
    new_qasm_line += apply_qasm_pauli_gate(qreg_name, control, c)

    return new_qasm_line


def pauli_twirl_cnots(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Pauli-twirl all CNOT-gates in a general quantum circuit. This function is included here for completeness.


    Parameters
    ----------
    qc : qiskit.QuantumCircuit
        The original quantum circuit.

    Returns
    -------
    pauli_twirled_qc : qiskit.QuantumCircuit
        The quantum circuit where all CNOT-gates have been Pauli-twirled.
    """

    # The circuit may be expressed in terms of various types of gates.
    # The 'Unroller' transpiler pass 'unrolls' the circuit to be expressed in terms of the
    # physical gate set [u1,u2,u3,cx]
    unroller = Unroller(["u1", "u2", "u3", "cx"])
    pm = PassManager(unroller)

    unrolled_qc = pm.run(qc)

    circuit_qasm = unrolled_qc.qasm()
    new_circuit_qasm_str = ""

    qreg_name = find_qreg_name(circuit_qasm)

    for i, line in enumerate(circuit_qasm.splitlines()):
        if line[0:2] == "cx":
            new_circuit_qasm_str += pauli_twirl_cnot_gate(qreg_name, line)
        else:
            new_circuit_qasm_str += line + "\n"

    new_qc = QuantumCircuit.from_qasm_str(new_circuit_qasm_str)

    # The "Optimize1qGates" transpiler pass optimizes chains of single-qubit gates by collapsing them into
    # a single, equivalent u3-gate

    # We want to avoid that the transpiler optimizes CNOT-gates, as the ancillary CNOT-gates must be kept
    # to keep the noise amplification

    optimize1qates = Optimize1qGates()
    pm = PassManager(optimize1qates)

    return pm.run(new_qc)