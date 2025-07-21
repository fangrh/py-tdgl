from typing import Union

import h5py


class Layer:
    """A superconducting thin film.

    Args:
        london_lambda: The London penetration depth of the film.
        coherence_length: The superconducting coherence length of the film.
        thickness: The thickness of the film.
        conductivity: The normal state conductivity of the superconductor in
            Siemens / length_unit.
        u: The ratio of the relaxation times for the order parameter amplitude
            and phase. This value is 5.79 for dirty superconductors.
        gamma: This parameter quantifies the effect of inelastic phonon-electron
            scattering. :math:`\\gamma` is proportional to the inelastic scattering
            time and the size of the superconducting gap.
        z0: Vertical location of the film.
        T_0: The dimensionless critical temperature for heat diffusion.
        kappa_eff: The effective thermal conductivity (dimensionless).
        eta: The heat exchange coefficient with environment (dimensionless).
        C_eff: The effective heat capacity (dimensionless).
        T_heat: The environment temperature (dimensionless).
    """

    def __init__(
        self,
        *,
        london_lambda: float,
        coherence_length: float,
        thickness: float,
        conductivity: Union[float, None] = None,
        u: float = 5.79,
        gamma: float = 10.0,
        z0: float = 0,
        # 热扩散参数
        T_0: Union[float, None] = None,
        kappa_eff: Union[float, None] = None,
        eta: Union[float, None] = None,
        C_eff: Union[float, None] = None,
        T_heat: Union[float, None] = None,
    ):
        self.london_lambda = london_lambda
        self.coherence_length = coherence_length
        self.thickness = thickness
        self.conductivity = conductivity
        self.u = u
        self.gamma = gamma
        self.z0 = z0
        # 热扩散参数
        self.T_0 = T_0
        self.kappa_eff = kappa_eff
        self.eta = eta
        self.C_eff = C_eff
        self.T_heat = T_heat

    @property
    def Lambda(self) -> float:
        """Effective magnetic penetration depth, :math:`\\Lambda=\\lambda^2/d`."""
        return self.london_lambda**2 / self.thickness

    def copy(self) -> "Layer":
        """Create a deep copy of the :class:`tdgl.Layer`."""
        return Layer(
            london_lambda=self.london_lambda,
            coherence_length=self.coherence_length,
            thickness=self.thickness,
            conductivity=self.conductivity,
            u=self.u,
            gamma=self.gamma,
            z0=self.z0,
            # 热扩散参数
            T_0=self.T_0,
            kappa_eff=self.kappa_eff,
            eta=self.eta,
            C_eff=self.C_eff,
            T_heat=self.T_heat,
        )

    def to_hdf5(self, h5_group: h5py.Group) -> None:
        """Save the :class:`tdgl.Layer` to an :class:`h5py.Group`.

        Args:
            h5_group: An open :class:`h5py.Group` to which to save the layer.
        """
        h5_group.attrs["london_lambda"] = self.london_lambda
        h5_group.attrs["coherence_length"] = self.coherence_length
        h5_group.attrs["thickness"] = self.thickness
        h5_group.attrs["u"] = self.u
        h5_group.attrs["gamma"] = self.gamma
        h5_group.attrs["z0"] = self.z0
        if self.conductivity is not None:
            h5_group.attrs["conductivity"] = self.conductivity
        # 保存热扩散参数
        if self.T_0 is not None:
            # 检查 T_0 是否为数组
            if hasattr(self.T_0, '__len__') and not isinstance(self.T_0, (str, bytes)):
                h5_group["T_0"] = self.T_0
            else:
                h5_group.attrs["T_0"] = self.T_0
        if self.kappa_eff is not None:
            # 检查 kappa_eff 是否为数组
            if hasattr(self.kappa_eff, '__len__') and not isinstance(self.kappa_eff, (str, bytes)):
                h5_group["kappa_eff"] = self.kappa_eff
            else:
                h5_group.attrs["kappa_eff"] = self.kappa_eff
        if self.eta is not None:
            # 检查 eta 是否为数组
            if hasattr(self.eta, '__len__') and not isinstance(self.eta, (str, bytes)):
                # 如果是数组，保存为数据集
                h5_group["eta"] = self.eta
            else:
                # 如果是标量，保存为属性
                h5_group.attrs["eta"] = self.eta
        if self.C_eff is not None:
            # 检查 C_eff 是否为数组
            if hasattr(self.C_eff, '__len__') and not isinstance(self.C_eff, (str, bytes)):
                h5_group["C_eff"] = self.C_eff
            else:
                h5_group.attrs["C_eff"] = self.C_eff
        if self.T_heat is not None:
            # 检查 T_heat 是否为数组
            if hasattr(self.T_heat, '__len__') and not isinstance(self.T_heat, (str, bytes)):
                h5_group["T_heat"] = self.T_heat
            else:
                h5_group.attrs["T_heat"] = self.T_heat

    @staticmethod
    def from_hdf5(h5_group: h5py.Group) -> "Layer":
        """Load a :class:`tdgl.Layer` from an :class:`h5py.Group`.

        Args:
            h5_group: An open :class:`h5py.Group` from which to load the layer.

        Returns:
            A new :class:`tdgl.Layer` instance.
        """

        def get(key, default=None):
            if key in h5_group.attrs:
                return h5_group.attrs[key]
            elif key in h5_group:  # 检查是否为数据集
                return h5_group[key][:]
            return default

        return Layer(
            london_lambda=get("london_lambda"),
            coherence_length=get("coherence_length"),
            thickness=get("thickness"),
            conductivity=get("conductivity"),
            u=get("u"),
            gamma=get("gamma"),
            z0=get("z0"),
            # 热扩散参数
            T_0=get("T_0"),
            kappa_eff=get("kappa_eff"),
            eta=get("eta"),
            C_eff=get("C_eff"),
            T_heat=get("T_heat"),
        )

    def __eq__(self, other):
        if self is other:
            return True

        if not isinstance(other, Layer):
            return False

        return (
            self.london_lambda == other.london_lambda
            and self.coherence_length == other.coherence_length
            and self.thickness == other.thickness
            and self.conductivity == other.conductivity
            and self.u == other.u
            and self.gamma == other.gamma
            and self.z0 == other.z0
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"london_lambda={self.london_lambda}, "
            f"coherence_length={self.coherence_length}, "
            f"thickness={self.thickness}, "
            f"conductivity={self.conductivity}, "
            f"u={self.u}, "
            f"gamma={self.gamma}, "
            f"z0={self.z0}"
            f")"
        )
