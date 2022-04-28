<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- PROJECT LOGO -->
<br />


  <h3 align="center">A Generic approach for urgency detection in mails</h3>

  



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
        <li><a href="#code structure">Folder Structure</a></li>
      </ul>
    </li>
    <li>
       <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
       </ul>
       <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
   
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project



Email is still one of the most popular ways to
communicate online. Every day, people spend a significant
amount of time sending and receiving emails in order to ex-
change information, handle projects, and plan activities. Previous
research has looked into several methods for increasing email
productivity. The task has primarily been framed as a supervised
learning problem, with models of varying degrees of complexity
proposed to classify an email message into a specified taxonomy
of intents or classifications. This paper proposes a supervised
learning methodology that intends to classify email messages into
one of three categories - very urgent, urgent, and not urgent.
### Built With

* [Python](https://python.com)
* [sklearn](https://scikit-learn.org/stable/)
* [nltk](https://www.nltk.org/)
* [tkinter](https://docs.python.org/3/library/tkinter.html)
* [imap_tools](https://github.com/ikvk/imap_tools)
* [tensorflow](https://www.tensorflow.org/)
* [keras](https://keras.io/)
* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)


## Folder structure
/*root<br/>
      &emsp;&emsp;|<br/>
     &emsp;&emsp;|---data/<br/>
            &emsp;&emsp;&emsp;&emsp;|---urgency.csv<br/>
    &emsp;&emsp;|---mail.py/<br/>
    &emsp;&emsp;|---infer.py/<br/>
    &emsp;&emsp;|---train.py/<br/>
    &emsp;&emsp;|---pickle_model.pkl/<br/>
    &emsp;&emsp;|---tfid.pkl/<br/>

			

      
 





### Prerequisites


  ```sh
      pip3 install -r requirements.txt
  ```





<!-- GETTING STARTED -->
## Getting Started
Train the model
```sh
    python3 train.py
  
  ```

Run the alert python code locally
```sh
  python3 mail.py
  ```




<!-- ROADMAP -->

            



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - O P Joy Jefferson - joy.jefferson10@gmail.com
</br>
Hemankith Reddy - hemankith@gmail.com
Kulachi Thapae - TBU

Project Link: https://github.com/jeff10joy/UrgencyDetector



<!-- ACKNOWLEDGEMENTS -->





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->



