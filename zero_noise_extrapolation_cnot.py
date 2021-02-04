from qiskit import QuantumCircuit, execute, Aer, transpile

from qiskit.result.result import Result
from qiskit.providers.aer.noise import NoiseModel

from qiskit.transpiler import PassManager, PassManagerConfig, CouplingMap
from qiskit.transpiler.passes import Unroller, Optimize1qGates
from qiskit.transpiler.preset_passmanagers.level3 import level_3_pass_manager

from numpy import asarray, ndarray, shape, zeros, empty, average, transpose, dot
from numpy.linalg import solve

from zero_noise_extrapolation import Richardson_extrapolate

import random, os, pickle

from typing import Callable

"""
-- ZERO NOISE EXTRAPOLATION for CNOT-gates --

This implemention does quantum error mitigation using the method of zero noise extrapolation, by amplifying noise in
a quantum circuit by a set of noise amplification factors, then using Richardson extrapolation to extrapolate the
expectation value to the zero-noise limit.

The noise amplified and mitigated is specifically noise in CNOT-gates. The noise is amplified by n amplification
factors c=1, 3, 5, ..., 2n + 1 by repeating each CNOT gates c times. E.g. c=1 corresponds to the bare circuit and
c=3 corresponds to each CNOT gate replaced by CNOT*CNOT*CNOT. Each CNOT acting on the same control- and target-qubits
as the original CNOT-gate.

As CNOT*CNOT = Id, the identity, in the noise-free case the amplified CNOT (eqaul to CNOT^c) have the same action as a
single CNOT. In the noise-afflicted case, the action will be close to that of a single CNOT for a weak noise, but with
the CNOT-noise applied c times to the qubit throughout the process.

"""


class ZeroNoiseExtrapolation:

    def __init__(self, qc: QuantumCircuit, exp_val_func: Callable, backend=None, exp_val_options: dict = None,
                 noise_model: NoiseModel = None, n_amp_factors: int = 3, shots: int = 8192,
                 pauli_twirl: bool = False,pass_manager: PassManager = None,
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
            quantum circuit. The function should take two arguments: a qiskit.result.result.Result object as its first,
            and a dictionary with possible options as its second.

        backend :
            A qiskit backend, either an IBMQ quantum backend or a simulator backend, for circuit executions.

        exp_val_options : dict, (optional)
            Options for the exp_val_func expectation value function

        noise_model : qiskit.providers.aer.noise.NoiseModel, (optional)
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

        # Set backend for circuit execution
        if backend == None:
            self.backend = Aer.get_backend("qasm_simulator")
            self.is_simulator = True
        else:
            self.backend = backend
            self.is_simulator = backend.configuration().simulator

        self.exp_val_func = exp_val_func

        self.noise_model = noise_model

        self.n_amp_factors = n_amp_factors
        self.noise_amplification_factors = asarray([(1 + 2 * i) for i in range(0, n_amp_factors)])

        self.pauli_twirl = pauli_twirl

        # Variables involved in saving of results to disk
        self.save_results, self.option = save_results, option
        self.experiment_name = ""
        self.set_experiment_name(experiment_name)
        if self.option == None:
            self.option = {}

        # Max number of shots for one circuit execution on IBMQ devices is 8192.
        # To do more shots, we have to partition them up into several executions.
        self.shots, self.repeats = self.partition_shots(shots)

        # Do an initial optimization of the quantum circuit. Either with a custom pass manager, or with the
        # optimization_level=3 transpiler preset (the heaviest optimization preset)
        circuit_read_from_file = False
        if self.save_results:
            qc_from_file = self.read_from_file(self.experiment_name + ".circuit")
            if qc_from_file != None:
                circuit_read_from_file = True
                self.qc = qc_from_file
        if not circuit_read_from_file:
            self.qc = self.transpile_circuit(qc, custom_pass_manager=pass_manager)

        # Initialization of variables for later use:

        self.counts = []

        self.depths, self.gamma_factors = empty(n_amp_factors), empty(n_amp_factors)

        self.exp_vals = zeros(0)
        self.all_exp_vals = zeros(0)
        self.noise_amplified_exp_vals = zeros(0)

        self.result = None

        self.measurement_results = []

    def partition_shots(self, shots: int) -> (int, int):
        """
        IBMQ devices limits circuit executions to a max of 8192 shots per experiment. To perform more than 8192 shots,
        the experiment has to be repeated.
        :param shots: Total number of shots of the circuit to be executed
        """
        if shots <= 8192:
            return shots, 1
        else:
            return int(shots / self.repeats), (shots // 8192) + 1

    def set_shots(self, shots: int):
        self.shots, self.repeats = self.partition_shots(shots)

    def get_shots(self):
        return self.repeats * self.shots

    def set_experiment_name(self, experiment_name):
        if self.save_results:
            if experiment_name == "":
                raise Exception("experiment_name cannot be empty when saving results")
            self.experiment_name = experiment_name
            self.experiment_name += "_ZNE_CNOTrep"
            self.experiment_name += "_backend" + self.backend.name()
            self.experiment_name += "_shots" + str(self.get_shots())
            self.experiment_name += "_paulitwirling" + str(self.pauli_twirl)

    def read_from_file(self, filename: str):
        directory = self.option.get("directory", "results")
        if os.path.isfile(directory + "/" + filename):
            file = open(directory + "/" + filename, "rb")
            data = pickle.load(file)
            file.close()
            return data
        else:
            return None

    def write_to_file(self, filename: str, data):
        directory = self.option.get("directory", "results")
        file = open(directory + "/" + filename)
        pickle.dump(data, file)

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
        Transpile and optimize the input circuit, optionally using a custom pass manager.
        If no custom pass manager is given, use the optimization_level 3 preset for the qiskit transpiler,
        which gives heaviest circuit optimization.

        As we want to add additional CNOTs for noise amplification and possibly additional single qubit gates
        for Pauli twirling, we need to transpile the circuit before both the noise amplification is applied and
        before circuit execution. This is to avoid the additional CNOT-gates beinng removed by the transpiler.

        The Optimize1qGates transpiler pass will be used later to optimize single qubit gates added during
        the Pauli-twirling, as well as the Unroller pass which merely decomposes the given circuit gates into
        the given set of basis gates.

        :return: The transpiled circuit
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

        # As there is some randomness involved in the qiskit transpiling, we might want to save
        # the specific transpiled circuit that is used
        if self.save_results:
            filename = self.experiment_name + ".circuit"
            self.write_to_file(filename, transpiled_circuit)

        return transpiled_circuit

    def execute_circuit(self, qc: QuantumCircuit, shots=None) -> Result:
        """

        Parameters
        ----------
        qc
        shots

        Returns
        -------

        """
        """
        Execute a quantum circuit for the specified amount of shots to obtain a set of measurement results.

        :param qc: Circuit to be executed
        :return: Measurement results as a qiskit.result.result.Result object
        """

        if shots == None:
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
        result

        Returns
        -------

        """
        experiment_exp_vals = zeros(len(result.results))
        for i, experiment_result in enumerate(result.results):
            experiment_exp_vals[i] = self.exp_val_func(experiment_result)
        return average(experiment_exp_vals), experiment_exp_vals

    def mitigate(self, verbose: bool = False) -> float:
        """
        Do error mitigation for general CNOT-noise in the given quantum circuit by zero-noise extrapolation.

        :param verbose: Do prints during the computation, True / False
        :return: The mitigated expectation value
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
            print("Constructing circuits")

        noise_amplified_circuits = []

        for j, amp_factor in enumerate(self.noise_amplification_factors):
            noise_amplified_circuits.append(self.noise_amplify_and_pauli_twirl_cnots(qc=self.qc, amp_factor=amp_factor,
                                                                                     pauli_twirl=self.pauli_twirl))
            self.depths[j] = noise_amplified_circuits[-1].depth()

        if verbose:
            print("Depths=", self.depths, sep="")

            print("Executing circuits")

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

            if not circuit_read_from_file:
                circuit_measurement_results = self.execute_circuit(noise_amplified_circuits[i])
                if self.save_results:
                    self.write_to_file(self.experiment_name + "_r{:].results".format(self.noise_amplification_factors[i]),
                                       circuit_measurement_results)

            self.noise_amplified_exp_vals[i], self.all_exp_vals[i, :] = self.compute_exp_val(
                circuit_measurement_results)

            self.measurement_results.append(circuit_measurement_results)

        self.result = Richardson_extrapolate(self.noise_amplified_exp_vals, self.noise_amplification_factors)[0]

        if verbose:
            print("-----", "\nError mitigation done",
                  "\nBare circuit expectation value: ", self.noise_amplified_exp_vals[0],
                  "\nMitigated expectation value:", self.result, "\n-----", sep="")

        return self.result


# PAULI TWIRLING AND NOISE AMPLIFICATION HELP FUNCTIONS

# Conversion from pauli x/y/z-gates to physical u1/u3-gates in correct OpenQASM-format
PHYSICAL_GATE_CONVERSION = {"X": "u3(pi,0,pi)", "Z": "u1(pi)", "Y": "u3(pi,pi/2,pi/2)"}


def find_qreg_name(circuit_qasm: str) -> str:
    """
    Finds the name of the quantum register in the circuit.

    :param circuit_qasm: OpenQASM string with instructions for the entire circuit
    :return: Name of the quantum register
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
    Find indices of control and target qubits for the CNOT-gate in question

    :param qasm_line: OpenQASM line containing the CNOT
    :return: Indices of control and target qubits
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

    :param control_in: Pauli gates on control qubit before CNOT
    :param target_in: Pauli gates on target qubit before CNOT
    :return: Equivalent Pauli gates on control and target qubits after CNOT
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
    Construct a OpenQASM-string with the instruction to apply the given pauli gates to
    the given qubit

    :param qreg_name: Name of quantum register
    :param qubit: Index of qubit
    :param pauli_gates: The Pauli gates to be applied
    :return: The OpenQASM-string with the instruction
    """
    new_qasm_line = ''
    for gate in pauli_gates:
        if gate != 'I':
            u_gate = PHYSICAL_GATE_CONVERSION[gate]
            new_qasm_line += u_gate + ' ' + qreg_name + '[' + str(qubit) + '];' + '\n'
    return new_qasm_line


def pauli_twirl_cnot_gate(qreg_name: str, qasm_line_cnot: str) -> str:
    """
    Pauli-twirl a CNOT-gate from the given OpenQASM string line containing the CNOT.
    This will look something like: cx q[0],q[1];

    :param qreg_name: Name of quantum register
    :param qasm_line_cnot: OpenQASM-line containing the CNOT to pauli twirl
    :return:
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
    General function for Pauli-twirling all CNOT-gates in a quantum circuit.
    Included for completeness.

    :param qc: quantum circuit for which to Pauli twirl all CNOT gates
    :return: Pauli twirled quantum circuit
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