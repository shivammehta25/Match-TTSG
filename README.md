# Match-TTSG: Unified speech and gesture synthesis using flow matching


<head>
  <link rel="icon" type="image/x-icon" href="favicon.ico">
  <meta name="msapplication-TileColor" content="#da532c">
  <meta charset="UTF-8">
  <meta name="theme-color" content="#ffffff">
  <meta property="og:title" content="Matcha-TTS: A fast TTS architecture with conditional flow matching" />
  <meta name="og:description" content="We propose Matcha-TTS, a new approach to non-autoregressive neural TTS, that uses conditional flow matching to speed up ODE-based speech synthesis. Our method is probabilistic, has compact memory footprint, sounds highly natural, is very fast to synthesise from">
  <meta property="og:image" content="images/architecture.png" />
  <meta property="twitter:image" content="images/architecture.png" />
  <meta property="og:type" content="website" />
  <meta property="og:site_name" content="Matcha-TTS" />
  <meta name="twitter:card" content="images/architecture.png" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="keywords" content="tts, text to speech, probabilistic machine learning, diffusion models, conditional flow matching, generative modelling, machine learning, deep learning, speech synthesis, research, phd">
  <meta name="description" content="We propose Matcha-TTS, a new approach to non-autoregressive neural TTS, that uses conditional flow matching to speed up ODE-based speech synthesis. Our method is probabilistic, has compact memory footprint, sounds highly natural, is very fast to synthesise from." />
</head>

##### [Shivam Mehta][shivam_profile], [Ruibo Tu][ruibo_profile], [Simon Alexanderson][simon_profile], [Jonas Beskow][jonas_profile], [Éva Székely][eva_profile], and [Gustav Eje Henter][gustav_profile]


We introduce a new method, _Match-TTSG_, for diffusion-like joint synthesis of speech and 3D gestures from text. Our main improvements are:
1. A new architecture that unifies speech and motion synthesis into one single pathway and decoder.
2. Training using [flow matching][lipman_et_al], a.k.a. [rectified flows][liu_et_al].

Compared to the previous state of the art, our new method:
- Improves speech and motion quality
- Is smaller
- Is 10 times faster
- Generates speech and gestures that are a much better fit for each other

To our knowledge, this is the first method synthesising 3D motion using flow matching or rectified flows.

Please check out the examples below and [read our arXiv preprint for more details][arxiv_link]. Code and pre-trained models will be made available in a few weeks.


[shivam_profile]: https://www.kth.se/profile/smehta
[ruibo_profile]: https://www.kth.se/profile/ruibo
[jonas_profile]: https://www.kth.se/profile/beskow
[eva_profile]: https://www.kth.se/profile/szekely
[simon_profile]: https://www.kth.se/profile/simonal
[gustav_profile]: https://people.kth.se/~ghe/
[this_page]: https://shivammehta25.github.io/Match-TTSG
[arxiv_link]: https://arxiv.org
[github_link]: https://github.com/shivammehta25/Match-TTSG
[lipman_et_al]: https://arxiv.org/abs/2210.02747
[liu_et_al]: https://arxiv.org/abs/2209.03003