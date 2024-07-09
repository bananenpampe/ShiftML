# Chemical shielding prediction from the SOAP model
The program predicts the chemical sheidling of a given chemical structure based on the ridge regression results of the SOAP descriptors.

To execute the program, first generate the tensor files storing the model coefficients by running the program `Fit_parameters.py`.
```Shell
cd model
python Fit_paraeterms.py
```
This conducts ridge regression on the SOAP descriptors, taking the chemical shielding results obtained from DFT calculations.

With the parater files generated, one can test the ASE calculator by running:
```Shell
cd ../example
python example.py
```
The detailed usage of the model is included in the file `example.py`.

To be specific, this file imports the modules as
```python
from ShiftML import ShiftML, StandardOutput
from ase.build import bulk
```
and loads the calculator along with an example diamond model by
```python
frame = bulk("C", "diamond", a=3.566)
calc = ShiftML()
frame.set_calculator(calc)
```
The outputs which contain the predicted chemical shielding is called by
```python
output = frame.calc.run_model(frame, StandardOutput)
```
and is stored in the ASE attributes by
```python
frame.arrays["cs_iso"] = output['mtt::cs_iso'].block(0).values.detach().numpy()
```