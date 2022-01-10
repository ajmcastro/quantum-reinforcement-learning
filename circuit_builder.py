from qiskit import *
from qiskit.circuit import Parameter

import numpy as np

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class CircuitBuilder(metaclass=Singleton):
    def __init__(self, backend):
        self.backend = backend

        self.u_parameters = {}
        self.parameters = {}

        self.compiled_circs = {}
        self.compiled_instrs = {}

    def get_u_parameters(self, num_qubits):
        if num_qubits in self.u_parameters:
            return self.u_parameters[num_qubits]

        self.u_parameters[num_qubits] = np.array([Parameter(f'U_θ{i + 1}') for i in range(2**num_qubits - 1)])

        return self.u_parameters[num_qubits]

    def get_parameters(self, num_qubits):
        if num_qubits in self.parameters:
            return self.parameters[num_qubits]
        
        self.parameters[num_qubits] = np.array([Parameter(f'θ{i + 1}') for i in range((2**num_qubits - 1) * 2**num_qubits * 2)])

        return self.parameters[num_qubits]

    def get_U(self, num_qubits, bound_parameters):
        circ = self._get_cached_circuit('U', num_qubits)
        parameters = self.get_u_parameters(num_qubits)
        return circ.assign_parameters(dict(zip(parameters, bound_parameters)))

    def get_Up(self, num_qubits, bound_parameters):
        circ = self._get_cached_circuit('Up', num_qubits)
        parameters, _ = np.split(self.get_parameters(num_qubits), 2)
        return circ.assign_parameters(dict(zip(parameters, bound_parameters)))

    def get_ARO(self, num_qubits, num_ancilla):
        return self._get_cached_instruction('ARO', num_qubits, num_ancilla)

    def _get_cached_circuit(self, circ_type, num_qubits):
        if (circ_type, num_qubits) in self.compiled_circs:
            return self.compiled_circs[(circ_type, num_qubits)]

        circ = None

        if circ_type == 'U':
            circ = self._U(num_qubits, self.get_u_parameters(num_qubits))
        elif circ_type == 'Up':
            circ = self._Up(num_qubits)
        elif circ_type == 'Vp':
            circ = self._Vp(num_qubits)
        elif circ_type == 'W':
            circ = self._W(num_qubits)

        compiled_circ = transpile(circ, backend=self.backend)
        self.compiled_circs[(circ_type, num_qubits)] = compiled_circ
        
        return compiled_circ

    def _get_cached_instruction(self, instr_type, num_qubits, num_ancilla):
        if (instr_type, num_qubits, num_ancilla) in self.compiled_instrs:
            return self.compiled_instrs[(instr_type, num_qubits, num_ancilla)]

        circ = None

        if instr_type == 'PD':
            circ = self._PD(num_qubits, num_ancilla)
        elif instr_type == 'ARO':
            circ = self._ARO(num_qubits, num_ancilla)

        compiled_instr = transpile(circ, backend=self.backend).to_instruction()
        self.compiled_instrs[(instr_type, num_qubits, num_ancilla)] = compiled_instr

        return compiled_instr

    def _U(self, num_qubits, params):
        circ = QuantumCircuit(num_qubits, name='U')

        circ.ry(params[0], num_qubits - 1)

        if num_qubits == 1:
            return circ
        elif num_qubits == 2:
            circ.cry(params[1], 1, 0, ctrl_state='0')
            circ.cry(params[2], 1, 0, ctrl_state='1')
        else:
            lhs, rhs = np.split(params[1:], 2)
            lhs_circ = self._U(num_qubits - 1, lhs).control(ctrl_state='0')
            rhs_circ = self._U(num_qubits - 1, rhs).control(ctrl_state='1')

            qubits = [num_qubits - 1] + list(range(num_qubits - 1))
            circ.compose(lhs_circ, qubits, inplace=True)
            circ.compose(rhs_circ, qubits, inplace=True)

        return circ

    def _Up(self, num_qubits):
        circ = QuantumCircuit(2 * num_qubits, name='Up')
        params, _ = np.split(self.get_parameters(num_qubits), 2)

        for i in range(2**num_qubits):
            ctrl_state = np.binary_repr(i, num_qubits)
            params_i = params[(2**num_qubits - 1) * i:(2**num_qubits - 1) * (i + 1)]

            Ui = self.get_U(num_qubits, params_i)
            circ.compose(Ui.control(num_ctrl_qubits=num_qubits, ctrl_state=ctrl_state), inplace=True)

        return circ

    def _Vp(self, num_qubits):
        circ = QuantumCircuit(2 * num_qubits, name='Vp')
        _, params = np.split(self.get_parameters(num_qubits), 2)

        circ.swap(list(range(num_qubits)), list(range(num_qubits, 2 * num_qubits)))

        for i in range(2**num_qubits):
            ctrl_state = np.binary_repr(i, num_qubits)
            params_i = params[(2**num_qubits - 1) * i:(2**num_qubits - 1) * (i + 1)]

            Ui = self.get_U(num_qubits, params_i)
            circ.compose(Ui.control(num_ctrl_qubits=num_qubits, ctrl_state=ctrl_state), inplace=True)

        circ.swap(list(range(num_qubits)), list(range(num_qubits, 2 * num_qubits)))

        return circ

    def _W(self, num_qubits):
        circ = QuantumCircuit(2 * num_qubits, name='W')

        Up = self._get_cached_circuit('Up', num_qubits)
        Vp = self._get_cached_circuit('Vp', num_qubits)

        # Ref A
        circ.compose(Up.inverse(), inplace=True)
        circ.x(list(range(num_qubits, 2 * num_qubits)))

        if num_qubits == 1:
            circ.z(1)
        else:
            circ.h(2 * num_qubits - 1)
            circ.mcx(list(range(num_qubits, 2 * num_qubits - 1)), 2 * num_qubits - 1)
            circ.h(2 * num_qubits - 1)
    
        circ.x(list(range(num_qubits, 2 * num_qubits)))
        circ.compose(Up, inplace=True)
        
        # Ref B
        circ.compose(Vp.inverse(), inplace=True)
        circ.x(list(range(num_qubits)))

        if num_qubits == 1:
            circ.z(0)
        else:
            circ.h(num_qubits - 1)
            circ.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            circ.h(num_qubits - 1)
    
        circ.x(list(range(num_qubits)))
        circ.compose(Vp, inplace=True)

        return circ

    def _PD(self, num_qubits, num_ancilla):
        anc = AncillaRegister(num_ancilla)
        qreg1 = QuantumRegister(num_qubits)
        qreg2 = QuantumRegister(num_qubits)
        circ = QuantumCircuit(anc, qreg1, qreg2, name='PD')

        W = self._get_cached_circuit('W', num_qubits)
        c_W = W.control().to_instruction()

        circ.h(anc)

        for i in range(num_ancilla):
            qubits = [anc[i]] + qreg1[:] + qreg2[:]
            circ.append(c_W, qubits)

        circ.h(anc)

        return circ

    def _ARO(self, num_qubits, num_ancilla):
        anc = AncillaRegister(num_ancilla)
        qreg1 = QuantumRegister(num_qubits)
        qreg2 = QuantumRegister(num_qubits)
        circ = QuantumCircuit(anc, qreg1, qreg2, name='ARO')

        PD = self._get_cached_instruction('PD', num_qubits, num_ancilla)

        circ.reset(anc)

        circ.append(PD, anc[:] + qreg1[:] + qreg2[:])
        circ.x(anc)

        if num_ancilla == 1:
            circ.z(anc[0])
        else:
            circ.h(anc[-1])
            circ.mcx(anc[:-1], anc[-1])
            circ.h(anc[-1])
    
        circ.x(anc)
        circ.append(PD.inverse(), anc[:] + qreg1[:] + qreg2[:])

        return circ