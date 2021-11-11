---
layout: post
title: "Linked List Project--Playing with Sound"
date: 2016-06-24
---

<iframe width="640" height="480" src="https://www.youtube.com/embed/vrChyNrIkZ4" frameborder="0" allowfullscreen></iframe><br>

It's an implement the following interface:<br>
MusicList.java<br>
import java.util.Iterator;<br>
public interface MusicList <br>
{<br>
    public int getNumChannels();<br>

 public float getSampleRate();<br>
 public int getNumSamples();<br>
 public float getDuration();<br>
 public void addEcho(float delay, float percent);<br>
 public void reverse();<br>
 public void changeSpeed(float percentChange);<br>
 public void changeSampleRate(float newRate);<br>
 public void addSample(float sample);<br>
 public void addSample(float sample[]);<br>
 public Iterator<float[]> iterator();<br>
 public Iterator<Float> iterator(int channel);<br>
 public void clip(float startTime, float duration);<br>
 public void spliceIn(float startSpliceTime, MusicList clipToSplice);<br>
 public void combine(MusicList clipToCombine, boolean allowClipping); <br>
        public void makeMono(boolean allowClipping);<br>
 public MusicList clone();<br>
}<br>


References:<br>

https://www.cs.usfca.edu/~galles/cs245/soundList/SoundListS16.html<br>
https://github.com/wchuanghard/wchuanghard.github.io/blob/master/files/SoundList%20Assignment.pdf<br>
