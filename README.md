



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![License][license-shield]][license-url]



# SecurityML



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Contributors](#contributors)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Citations](#citations)


<!-- ABOUT THE PROJECT -->
## About The Project

This is a university project developed for the "Information Security" course at the University of Deusto, for this project we aimed to explore the application of ML techniques to Computer Security.


Publication (September, 2021):  [''Deep Learning Applications on Cybersecurity''](https://link-springer-com.focus.lib.kth.se/chapter/10.1007/978-3-030-86271-8_51) was accepted by HAIS 2021.

### Built With

* [Python](https://www.python.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [Streamlit](https://streamlit.io/)


## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
<table align="center">
  <tr>
    <td align="center"><a href="https://github.com/carloslago">
        <img src="https://avatars2.githubusercontent.com/u/15263623?s=400&v=4" 
        width="150px;" alt="Carlos Lago"/><br/><sub><b>Carlos Lago</b></sub></a><br/></td>
    <td align="center"><a href="https://github.com/rafaelromon">
        <img src="https://avatars0.githubusercontent.com/u/15263554?s=400&v=4" 
        width="150px;" alt="Rafael Romón"/><br /><sub><b>Rafael Romón</b></sub></a><br/></td>
  </tr>
</table>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites
* Python
```sh
sudo apt install python3 python3-dev
```

* Tensorflow and Keras

  To install tensorflow and keras you should try following an official guide, as it is fairly complicated.


### Installation
 
1. Clone the repo
```sh
git clone https://github.com/rafaelromon/SecurityML
```
2. Install Python packages
```sh
sudo pip install -r requirements.txt
```
<!-- USAGE EXAMPLES -->
## Usage

For the most part this project has packages aimed to develop ML models, but you can run a demo powered by streamlit to test the models out. 

```sh
cd steamlit_web; streamlit run app.py
```

please note that the NSFW classification demo doesn't work unless you previously train a model, because of the 250mb size limit of github we were unable to upload ours.

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

<!-- LICENSE -->
## License

Distributed under the GPL License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

If you are not that tech savvy feel free to send us any bug reports, ask me any questions or request any features via email, just keep in mind we did this as a university project.

## Citations
Feel free to cite the following publication:

```
 @article{lago_romón_lópez_urquijo_tellaeche_bringas_2021, title={Deep learning applications on cybersecurity}, DOI={10.1007/978-3-030-86271-8_51}, journal={Lecture Notes in Computer Science}, author={Lago, Carlos and Romón, Rafael and López, Iker Pastor and Urquijo, Borja Sanz and Tellaeche, Alberto and Bringas, Pablo García}, year={2021}, pages={611–621}} 
```



[license-shield]: https://img.shields.io/github/license/rafaelromon/SecurityML
[license-url]: https://github.com/rafaelromon/SecurityML/blob/master/LICENSE
