# EHSS voltage-driven model

## Run

Python 3.14 is the tested environment.

```sh
python -m pip install -r requirements.txt
python model.py
```

The run saves `five_outputs.png` in this folder. The first run compiles the
numerical functions and can take longer.

## Files

- `model.py`: model, fixed initial conditions, and plotting.
- `input_voltage.csv`: `time_s,voltage_V`; 60,000 voltage samples at 1000 Hz
  for the supplied 60 s, 1 Hz sine-voltage case.
- `five_outputs.png`: five simulated channels, with the full record on the left
  and the final 5 s on the right.
- `requirements.txt`: Python dependencies.

The CSV voltage is applied directly in volts, with no voltage conversion in
the model. Each value is held for 0.001 s. The model evolves continuously from
fixed initial conditions without measured-output feedback or state resets.
The internal RK4 step is at most 0.00005 s and may decrease for stability;
input and output sampling remain 1000 Hz.

## Outputs

```python
from model import simulate
data = simulate(voltage)
```

`voltage` is a one-dimensional array in volts, sampled at 1000 Hz.
The returned array has one row per voltage sample and these columns:

`time_s, i_A, xs_m, pA_Pa, pB_Pa, xc_m`

Pressure A is the rod-end chamber; pressure B is the cap-end chamber.
The figure displays current in A, spool displacement in mm, pressures in bar,
and actuator displacement in cm. These display conversions do not change
the SI model calculations. No comparison data or error metrics are included.
