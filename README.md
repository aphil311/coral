<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<div align="center">
<!-- add image -->
<img src="./readme_assets/talos.png" alt="Logo">
<!-- <h3 align="center">TALOS</h3> -->

  <!-- <p align="center">
    A framework for creating robust, modularly aligned LLMs.
    <br />
    <a href="https://github.com/aphil311/talos"><strong>View progress »</strong></a>

  </p> -->

</div>

# TALOS: Tri-state aligned loyalty optimizations

[![CLicense](https://img.shields.io/badge/License%20-%20MIT%20-%20%23ff6863?style=flat)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE) [![Python 3.11](https://img.shields.io/badge/Python%20-%203.11%20-%20?style=flat&logo=python&logoColor=white)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Issues](https://img.shields.io/github/issues/aphil311/talos?style=flat&logo=github&logoColor=white)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)


<!-- ABOUT THE PROJECT -->
## About The Project
TALOS is a framework designed to enforce alignment and loyalty in large language models (LLMs) by integrating new reinforcement learning techniques with modular safety alignment strategies. It builds on concepts from Open, Monetizable, Loyal (OML) AI, Constitutional AI (CAI), Group Relative Policy Optimization (GRPO), Lazy Safety Alignment (Lisa), and Self-Instruct.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started


### Installation
1. Clone this repository with `git clone https://github.com/aphil311/talos.git`.
2. Install the dependencies with `pip install -r requirements.txt`.


### Usage 
1. Set the environment variable `OPENAI_API_KEY` to your OpenAI key (see example file).
2. Write your constitution and optionally seed prompts in `alignment_data` (see examples)
3. Run `python sl-cai/batched_data_generation {constitution} {num_examples}` to generate synthetic data.


<p align="right">(<a href="#readme-top">back to top</a>)</p> 



<!-- ROADMAP -->
## Roadmap

- [ ] **Supervised learning stage**
- [ ] **Reinforcement learning stage**
- [ ] **Fingerprinting**
- [ ] **Robust finetuning**


See the [open issues](https://github.com/aphil311/talos/issues) for a full list of proposed features (and known issues).



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
I would like to thank Professor Pramod Viswanath and Creston Brooks for their invaluable guidance, feedback, and support throughout the development of this paper.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/aphil311/hudson_food_finder.svg?style=for-the-badge
[contributors-url]: https://github.com/aphil311/hudson_food_finder/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/aphil311/hudson_food_finder.svg?style=for-the-badge
[forks-url]: https://github.com/aphil311/hudson_food_finder/network/members
[stars-shield]: https://img.shields.io/github/stars/aphil311/hudson_food_finder.svg?style=for-the-badge
[stars-url]: https://github.com/aphil311/hudson_food_finder/stargazers
[issues-shield]: https://img.shields.io/github/issues/aphil311/hudson_food_finder.svg?style=for-the-badge
[issues-url]: https://github.com/aphil311/hudson_food_finder/issues
[license-shield]: https://img.shields.io/github/license/aphil311/hudson_food_finder.svg?style=for-the-badge
[license-url]: https://github.com/aphil311/hudson_food_finder/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png

[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[Flask.com]: https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/en/2.2.x/
[Jinja.com]: https://img.shields.io/badge/jinja-cccccc.svg?style=for-the-badge&logo=jinja&logoColor=black
[Jinja-url]: https://jinja.palletsprojects.com/en/3.1.x/
[Render.com]: https://img.shields.io/badge/Render-%46E3B7.svg?style=for-the-badge&logo=render&logoColor=white
[Render-url]: https://render.com
[AWS.com]:https://img.shields.io/badge/Amazon_AWS-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white
[AWS-url]: https://aws.amazon.com


[PostgreSQL.com]: https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white
[PostgreSQL-url]: https://www.postgresql.org

