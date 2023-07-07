# Isomorphic linked Yagis

This project was inspired by the [Seattle Radio Field Day](https://www.seattleradiofieldday.org), which is a yearly event where the three local amateur radio groups set up a temporary communications station in a field, and makes as many contacts as possible over the course of a weekend.


## Antennas for Field Day

For a Field Day setup, it's hard to beat the simplicity of a [linked wire dipole](https://www.sotabeams.co.uk/linked-dipoles). These antennas are resonant on multiple bands, and easy to set up, requiring only three masts deployed in a row : two for each end, and one as a central support. 

Dipoles, unfortunately, are not directional antennas, and exhibit little gain in the direction in which we wish to communicate. Beam antennas are much more directional, and have a higher gain in the direction in which they are pointed; however, they make for difficult Field Day antennas. It's not easy to put up a tower robust enough to support a traditional, rigid beam antenna such as a [Yagi](https://en.wikipedia.org/wiki/Yagi%E2%80%93Uda_antenna), and the wire versions of Yagis are difficult to deploy, requiring many more masts than a linked dipole. These masts need to be spaced accurately, and the antenna's performance depends greatly on the accuracy of the spacing.




## Isomorphic Yagis

In order to address the problems caused by requiring many more masts, and in deploying them accurately, we propose a new type of Yagi configuration : the *isomorphic* Yagi. These two-element Yagis require only one additional mast, to support the central part of a V-shaped reflector, the ends of which are connected to the end-support masts of the driven element.

Designing an isomorphic Yagi is relatively simple. The dimensions to optimise over are :
- the spacing between the end-support masts of the driven element, $L$
- the length of the driven element, $d$, as a fraction of $L$
- the offset of the reflector's central support from the driven element, $D$
- the length of the reflector element, $r$, as a fraction of $D$
- - the height of the driven element above ground, $H$

TODO :
- add diagram
- get values for $L$, $d$, $D$, $r$ for 20m

We assume that wire elements are held up by rope, so that $d$ and $r$ can be adjusted by having more or less rope on either side of the driven element, and having the rope attach to the masts. This allows us to adjust the spacing between the reflector and the driven element (by increasing $D$ and reducing $r$ to compensate for the reflector's length); this also affects the angle of the bend in the V-shaped reflector.


## Multi-band isomorphic Yagis

The real challenge is in designing a linked isomorphic Yagi. We want to maintain the simplicity of the design goal : having a single added support for the reflector elements. As such, we need a linked reflector as well as a linked driven element, and the antenna's physical support parameters (the spacing between the driven element masts, and the offset of the reflector's central support) must be the same for all bands, such that a multi-band isomorphic Yagi antenna can be set up with four masts, in comparison with the simplest linked dipole's three masts.

For an antenna resonant on 80m, 40m, 20m, 15m, and 10m, we need to optimise over the following parameters :
- the spacing between the end-support masts of the driven element, $L$
- the lengths of the driven element, $d_{80}$, $d_{40}$, $d_{20}$, $d_{15}$, and $d_{10}$, as fractions of $L$
- the offset of the reflector's central support from the driven element, $D$
- the lengths of the reflector elements, $r_{80}$, $r_{40}$, $r_{20}$, $r_{15}$, and $r_{10}$, as fractions of $D$
- the height of the driven element above ground, $H$


## Designing a linked isomorphic Yagi

### Antenna evaluation

We use the [`nec2++`](https://github.com/tmolteno/necpp) implementation of the [NEC](https://en.wikipedia.org/wiki/Numerical_Electromagnetics_Code), driven by a Python script, to evaluate the performance of an antenna. The Python script generates the card-deck-style input file, runs `nec2++`, and parses the output to extract the antenna's gain and SWR. The antenna's performance is then calculated from these values.


### Algorithm

We use a [differential evolution algorithm](https://en.wikipedia.org/wiki/Differential_evolution) to optimise the parameters of the antenna. The algorithm is initialised with a population of 150 random antenna configurations satisfying the constraints below. The algorithm then proceeds with a crossover probability of 0.7 and a differential weight of 0.8. At each generation, antennas are mutated and, if the mutated antenna is better than the original, it replaces the original in the population. The algorithm terminates after 100 generations.


### Loss function

For a given band, an antenna's performance is the ratio of the antenna's highest gain between 45 and 75 degrees elevation, to its SWR at the band's centre frequency. The antenna's overall performance is the harmonic mean of its performance on each band.


### Constraints 

With the Seattle Radio Field Day in mind, we want to fix $L$ to about 21m or 69 feet, because this is twice the distance between masts in our typical seven-mast setup, to allow for full 80m antennas to be deployed. This limits the length of the maximum element to 69 feet, when $d_{80} = 1$, which should be plenty. We'll also fix the height $H$ to 10m, which is the height of the masts we use.

We impose constraints on the anchor offset, $D$, to be between 5m and 30m.

The driven element and reflector length ratios are constrained to be between :
 - 0.6 and 1 for 80m
 - 0.3 and 0.8 for 40m
 - 0.1 and 0.4 for 20m
 - 0.05 and 0.4 for 15m
 - 0.05 and 0.3 for 10m

Any antenna element that does not satisfy these constraints is clipped to these ranges.


## Improvements

- Rather than accepting only the best of the old-vs-mutated antenna pairs, use [tournament selection](https://en.wikipedia.org/wiki/Tournament_selection) to select stochastically.
- Include front-to-back ratio as part of the loss function.
- Test different values for crossover probability and differential weight.