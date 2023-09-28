# Match-TTSG: Unified speech and gesture synthesis using flow matching

<head>
  <link rel="icon" type="image/x-icon" href="favicon.ico">
  <meta name="msapplication-TileColor" content="#da532c">
  <meta charset="UTF-8">
  <meta name="theme-color" content="#ffffff">
  <meta property="og:title" content="Match-TTSG: Unified speech and gesture synthesis using flow matching" />
  <meta name="og:description" content="We introduce a new method, Match-TTSG, for diffusion-like joint synthesis of speech and 3D gestures from text.">
  <meta property="og:image" content="images/architecture.png" />
  <meta property="twitter:image" content="images/architecture.png" />
  <meta property="og:type" content="website" />
  <meta property="og:site_name" content="Match-TTSG: Unified speech and gesture synthesis using flow matching" />
  <meta property="og:url" content="https://shivammehta25.github.io/Match-TTSG/" />
  <meta name="twitter:card" content="images/architecture.png" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="keywords" content="tts, text to speech, probabilistic machine learning, diffusion models, conditional flow matching, generative modelling, machine learning, deep learning, speech synthesis, research, phd, gesture synthesis, multimodal synthesis">
  <meta name="description" content="We introduce a new method, Match-TTSG, for diffusion-like joint synthesis of speech and 3D gestures from text." />
</head>

##### [Shivam Mehta][shivam_profile], [Ruibo Tu][ruibo_profile], [Simon Alexanderson][simon_profile], [Jonas Beskow][jonas_profile], [Éva Székely][eva_profile], and [Gustav Eje Henter][gustav_profile]

We introduce a new method, _Match-TTSG_, for diffusion-like joint synthesis of speech and 3D gestures from text. Our main improvements are:

1. A new architecture that unifies speech and motion synthesis into one single pathway and decoder.
2. Training using [flow matching][lipman_et_al], a.k.a. [rectified flows][liu_et_al].

Compared to [the previous state of the art][diff_ttsg_link], our new method:

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
[diff_ttsg_link]: https://arxiv.org/abs/2306.09417

<style type="text/css">
    .tg {
    border-collapse: collapse;
    border-color: #9ABAD9;
    border-spacing: 0;
  }

  .tg td {
    background-color: #EBF5FF;
    border-color: #9ABAD9;
    border-style: solid;
    border-width: 1px;
    color: #444;
    font-family: Arial, sans-serif;
    font-size: 14px;
    overflow: hidden;
    padding: 0px 20px;
    word-break: normal;
    font-weight: bold;
    vertical-align: middle;
    horizontal-align: center;
    white-space: nowrap;
  }

  .tg th {
    background-color: #409cff;
    border-color: #9ABAD9;
    border-style: solid;
    border-width: 1px;
    color: #fff;
    font-family: Arial, sans-serif;
    font-size: 14px;
    font-weight: normal;
    overflow: hidden;
    padding: 0px 20px;
    word-break: normal;
    font-weight: bold;
    vertical-align: middle;
    horizontal-align: center;
    white-space: nowrap;
    padding: 10px;
    margin: auto;
  }

  .tg .tg-0pky {
    border-color: inherit;
    text-align: center;
    vertical-align: top,
  }

  .tg .tg-fymr {
    border-color: inherit;
    font-weight: bold;
    text-align: center;
    vertical-align: top
  }
  .slider {
  -webkit-appearance: none;
  width: 75%;
  height: 15px;
  border-radius: 5px;
  background: #d3d3d3;
  outline: none;
  opacity: 0.7;
  -webkit-transition: .2s;
  transition: opacity .2s;
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 25px;
  height: 25px;
  border-radius: 50%;
  background: #409cff;
  cursor: pointer;
}

.slider::-moz-range-thumb {
  width: 25px;
  height: 25px;
  border-radius: 50%;
  background: #409cff;
  cursor: pointer;
}

/* audio {
    width: 240px;
} */

/* CSS */
.button-12 {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 10px 54px;
  font-family: -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif;
  font-weight: bold;
  border-radius: 6px;
  border: none;

  background: #6E6D70;
  box-shadow: 0px 0.5px 1px rgba(0, 0, 0, 0.1), inset 0px 0.5px 0.5px rgba(255, 255, 255, 0.5), 0px 0px 0px 0.5px rgba(0, 0, 0, 0.12);
  color: #DFDEDF;
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
}

.button-12:focus {
  box-shadow: inset 0px 0.8px 0px -0.25px rgba(255, 255, 255, 0.2), 0px 0.5px 1px rgba(0, 0, 0, 0.1), 0px 0px 0px 3.5px rgba(58, 108, 217, 0.5);
  outline: 0;
}

video {
  margin: 1em;
}

audio {
  margin: 0.5em;
}

td img {
  position: relative;
  margin: 0 auto;
  max-width: 650px;
  padding: 5px;
  border: 0px;
}
</style>

<script>

  transcript_audio_only = {
    1: "I mean it it's not that I'm against it it's just that I just don't have the time and I just sometimes I'm not bothered and that sort of stuff.",
    2: "And then a few weeks later after that my parents were away my granny was minding us and again I don't know why I told my brother to do this but I was like here.",
    3: "But I remember once my parents were just downstairs in the kitchen and this is when mobile phones just began coming out. So, like my oldest brother and my oldest sister had a mobile phone each I'm pretty sure.",
    4: "If you like touched it, it was excruciatingly sore. And I went up to the teachers I was like look I'm after like really damaging my finger I might have to go to the doctors."
  }

  function play_audio(filename, audio_id,  condition_name, transcription){

      audio = document.getElementById(audio_id);
      audio_source = document.getElementById(audio_id + "-src");
      block_quote = document.getElementById(audio_id + "-transcript");
      stimulus_span = document.getElementById(audio_id + "-span");

      audio.pause();
      audio_source.src = filename;
      block_quote.innerHTML = transcription;
      stimulus_span.innerHTML = condition_name;
      audio.load();
      audio.play();
  }

</script>

## Stimuli from the evaluation test

### Speech-only evaluation

> Click the buttons in the table to load and play the different stimuli.

Currently loaded stimulus: <span id="audio-stimuli-from-listening-test-span" style="font-weight: bold;"> MAT-50 </span>

<p>Audio player: </p>
  <audio id="audio-stimuli-from-listening-test" controls>
    <source id="audio-stimuli-from-listening-test-src" src="stimuli/audio-only/MAT_50_C4_3_eval_0092.wav" type="audio/wav">
  </audio>

<p> Transcription: </p>
<blockquote style="height: 100px">
  <p id="audio-stimuli-from-listening-test-transcript">
    I mean it it's not that I'm against it it's just that I just don't have the time and I just sometimes I'm not bothered and that sort of stuff.
  </p>
</blockquote>

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Text prompt #</th>
      <th class="tg-0pky">NAT</th>
      <th class="tg-0pky">DIFF</th>
      <th class="tg-0pky" colspan="2">MAT</th>
      <th class="tg-0pky" colspan="2">SM</th>
    </tr>
    <tr>
      <th class="tg-0pky">Solver steps</th>
      <th class="tg-0pky">-</th>
      <th class="tg-0pky">50 & 500</th>
      <th class="tg-0pky">50</th>
      <th class="tg-0pky">500</th>
      <th class="tg-0pky">50</th>
      <th class="tg-0pky">500</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>1</td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/NAT_C4_3_eval_0092.wav', 'audio-stimuli-from-listening-test', 'NAT , Sentence 1', transcript_audio_only[1])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/DIFF_C4_3_eval_0092.wav', 'audio-stimuli-from-listening-test', 'DIFF , Sentence 1', transcript_audio_only[1])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/MAT_50_C4_3_eval_0092.wav', 'audio-stimuli-from-listening-test', 'MAT-50 , Sentence 1', transcript_audio_only[1])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/MAT_500_C4_3_eval_0092.wav', 'audio-stimuli-from-listening-test', 'MAT-500 , Sentence 1', transcript_audio_only[1])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/SM_50_C4_3_eval_0092.wav', 'audio-stimuli-from-listening-test', 'SM-50 , Sentence 1', transcript_audio_only[1])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/SM_500_C4_3_eval_0092.wav', 'audio-stimuli-from-listening-test', 'SM-500 , Sentence 1', transcript_audio_only[1])"/>
        </td>
    </tr>
    <tr>
        <td>2</td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/NAT_C3_7_eval_0163.wav', 'audio-stimuli-from-listening-test', 'NAT , Sentence 2', transcript_audio_only[2])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/DIFF_C3_7_eval_0163.wav', 'audio-stimuli-from-listening-test', 'DIFF , Sentence 2', transcript_audio_only[2])"/>
        </td> 
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/MAT_50_C3_7_eval_0163.wav', 'audio-stimuli-from-listening-test', 'MAT-50 , Sentence 2', transcript_audio_only[2])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/MAT_500_C3_7_eval_0163.wav', 'audio-stimuli-from-listening-test', 'MAT-500 , Sentence 2', transcript_audio_only[2])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/SM_50_C3_7_eval_0163.wav', 'audio-stimuli-from-listening-test', 'SM-50 , Sentence 2', transcript_audio_only[2])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/SM_500_C3_7_eval_0163.wav', 'audio-stimuli-from-listening-test', 'SM-500 , Sentence 2', transcript_audio_only[2])"/>
        </td>
    </tr>
    <tr>
        <td>3</td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/NAT_C3_7_eval_0047.wav', 'audio-stimuli-from-listening-test', 'NAT , Sentence 3', transcript_audio_only[3])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/DIFF_C3_7_eval_0047.wav', 'audio-stimuli-from-listening-test', 'DIFF , Sentence 3', transcript_audio_only[3])"/>
        </td> 
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/MAT_50_C3_7_eval_0047.wav', 'audio-stimuli-from-listening-test', 'MAT-50 , Sentence 3', transcript_audio_only[3])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/MAT_500_C3_7_eval_0047.wav', 'audio-stimuli-from-listening-test', 'MAT-500 , Sentence 3', transcript_audio_only[3])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/SM_50_C3_7_eval_0047.wav', 'audio-stimuli-from-listening-test', 'SM-50 , Sentence 3', transcript_audio_only[3])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/SM_500_C3_7_eval_0047.wav', 'audio-stimuli-from-listening-test', 'SM-500 , Sentence 3', transcript_audio_only[3])"/>
        </td>
    </tr>
    <tr>
        <td>4</td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/NAT_C3_7_eval_0447.wav', 'audio-stimuli-from-listening-test', 'NAT , Sentence 4', transcript_audio_only[4])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/DIFF_C3_7_eval_0447.wav', 'audio-stimuli-from-listening-test', 'DIFF , Sentence 4', transcript_audio_only[4])"/>
        </td> 
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/MAT_50_C3_7_eval_0447.wav', 'audio-stimuli-from-listening-test', 'MAT-50 , Sentence 4', transcript_audio_only[4])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/MAT_500_C3_7_eval_0447.wav', 'audio-stimuli-from-listening-test', 'MAT-500 , Sentence 4', transcript_audio_only[4])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/SM_50_C3_7_eval_0447.wav', 'audio-stimuli-from-listening-test', 'SM-50 , Sentence 4', transcript_audio_only[4])"/>
        </td>
        <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_audio('stimuli/audio-only/SM_500_C3_7_eval_0447.wav', 'audio-stimuli-from-listening-test', 'SM-500 , Sentence 4', transcript_audio_only[4])"/>
        </td>
    </tr>
  </tbody>
</table>

### Gesture-only evaluation (no audio)

<video id="gesture-only-video" class="video-js" controls width="640" height="360">
    <source id="gesture-only-video-source" src="stimuli/gesture-only/MAT_50_C4_3_eval_0092.mp4" type='video/mp4' />
</video>

Currently loaded: <span id="playing-gesture-only" style="font-weight: bold;" > MAT-50 1</span>

<blockquote style="height: 100px">
  <p id="gesture-only-transcription">
    If you like touched it, it was excruciatingly sore. And I went up to the teachers I was like look I'm after like really damaging my finger I might have to go to the doctors.
  </p>
</blockquote>

<p style="height: 10px">
    <span style="color: #ee4444; font-weight: bold" id="sm-50-trigger"> </div>
</p>

<script>
  gesture_only_video = document.getElementById('gesture-only-video')
  gesture_only_video_source = document.getElementById('gesture-only-video-source')
  gesture_only_span_text =  document.getElementById('playing-gesture-only')
  gesture_only_transcript = document.getElementById('gesture-only-transcription')

  trigger_span = document.getElementById('sm-50-trigger')

  function play_video(filename, text, trigger=false){
      id = text[text.length - 1];

      gesture_only_video.pause();
      gesture_only_video_source.src = filename;
      gesture_only_span_text.innerHTML = text;
      gesture_only_transcript.innerHTML = transcript_audio_only[id];
      gesture_only_video.load();
      gesture_only_video.play();

      if (trigger){
        trigger_span.innerHTML = "Note: SM-50 was excluded from this evaluations due to its low motion quality ";
      } else {
        trigger_span.innerHTML = "";
      }

  }
</script>

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Text prompt #</th>
      <th class="tg-0pky">NAT</th>
      <th class="tg-0pky">DIFF</th>
      <th class="tg-0pky" colspan="2">MAT</th>
      <th class="tg-0pky" colspan="2">SM</th>
    </tr>
    <tr>
      <th class="tg-0pky">Solver steps</th>
      <th class="tg-0pky">-</th>
      <th class="tg-0pky">50 & 500</th>
      <th class="tg-0pky">50</th>
      <th class="tg-0pky">500</th>
      <th class="tg-0pky">50</th>
      <th class="tg-0pky">500</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/NAT_C4_3_eval_0092.mp4', 'NAT 1')"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/DIFF_C4_3_eval_0092.mp4', 'DIFF 1')"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/MAT_50_C4_3_eval_0092.mp4', 'MAT-50 1')"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/MAT_500_C4_3_eval_0092.mp4', 'MAT-500 1')"/>
      </td>
      <td>
          <img src="images/play_button_red.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/SM_50_C4_3_eval_0092.mp4', 'SM-50 1', true)"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/SM_500_C4_3_eval_0092.mp4', 'SM-500 1')"/>
      </td>
    <tr>
      <td>2</td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/NAT_C3_7_eval_0163.mp4', 'NAT 2')"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/DIFF_C3_7_eval_0163.mp4', 'DIFF 2')"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/MAT_50_C3_7_eval_0163.mp4', 'MAT-50 2')"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/MAT_500_C3_7_eval_0163.mp4', 'MAT-500 2')"/>
      </td>
      <td>
          <img src="images/play_button_red.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/SM_50_C3_7_eval_0163.mp4', 'SM-50 2', true)"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/SM_500_C3_7_eval_0163.mp4', 'SM-500 2')"/>
      </td>
    <tr> 
    <tr>
      <td>3</td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/NAT_C3_7_eval_0047.mp4', 'NAT 3')"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/DIFF_C3_7_eval_0047.mp4', 'DIFF 3')"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/MAT_50_C3_7_eval_0047.mp4', 'MAT-50 3')"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/MAT_500_C3_7_eval_0047.mp4', 'MAT-500 3')"/>
      </td>
      <td>
          <img src="images/play_button_red.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/SM_50_C3_7_eval_0047.mp4', 'SM-50 3', true)"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/SM_500_C3_7_eval_0047.mp4', 'SM-500 3')"/>
      </td>
    <tr> 
    <tr>
      <td>4</td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/NAT_C3_7_eval_0447.mp4', 'NAT 4')"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/DIFF_C3_7_eval_0447.mp4', 'DIFF 4')"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/MAT_50_C3_7_eval_0447.mp4', 'MAT-50 4')"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/MAT_500_C3_7_eval_0447.mp4', 'MAT-500 4')"/>
      </td>
      <td>
          <img src="images/play_button_red.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/SM_50_C3_7_eval_0447.mp4', 'SM-50 4', true)"/>
      </td>
      <td>
          <img src="images/play_button.png" height=40 style="cursor: pointer;" onclick="play_video('stimuli/gesture-only/SM_500_C3_7_eval_0447.mp4', 'SM-500 4')"/>
      </td>
    <tr> 
    
    
  </tbody>
</table>
