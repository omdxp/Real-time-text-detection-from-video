using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace _18_Real_time_Text_Detection_from_Videos
{
    public partial class Form1 : Form
    {
        VideoCapture capture;
        bool pause = false;
        public Form1()
        {
            InitializeComponent();
        }

        private void openToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                capture = new VideoCapture(ofd.FileName);
                Mat m = new Mat();
                capture.Read(m);
                pictureBox1.Image = m.Bitmap;
            }
        }

        private async void playToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (capture == null)
                return;

            try
            {
                while (!pause)
                {
                    Mat m = new Mat();
                    capture.Read(m);

                    if (!m.IsEmpty)
                    {
                        pictureBox2.Image = m.Bitmap;
                        DetectText(m.ToImage<Bgr, byte>());
                        double fps = capture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.Fps);
                        await Task.Delay(1000 / Convert.ToInt32(fps));
                    }
                    else
                    {
                        break;
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void pauseToolStripMenuItem_Click(object sender, EventArgs e)
        {
            pause = !pause;
        }
        /// <summary>
        /// This method is for text detection
        /// </summary>
        /// <param name="img"></param>
        private void DetectText(Image<Bgr, byte> img)
        {
            /*
             * 1. Edge detection (sobel)
             * 2. Dilation (10, 1)
             * 3. FindContours
             * 4. Geometrical Constraints 
             */
            // Sobel
            Image<Gray, byte> sobel = img.Convert<Gray, byte>().Sobel(1, 0, 3)
                .AbsDiff(new Gray(0.0)).Convert<Gray, byte>()
                .ThresholdBinary(new Gray(100), new Gray(255));

            // Dilation
            Mat SE = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(10, 1),
                new Point(-1, -1));

            sobel = sobel.MorphologyEx(Emgu.CV.CvEnum.MorphOp.Dilate,
                SE,
                new Point(-1, -1),
                1,
                Emgu.CV.CvEnum.BorderType.Reflect,
                new MCvScalar(255));

            // FindContours
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat hier = new Mat();

            CvInvoke.FindContours(sobel, contours,
                hier,
                Emgu.CV.CvEnum.RetrType.External,
                Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);

            // Geometrical constraints
            List<Rectangle> list = new List<Rectangle>();

            for (int i = 0; i < contours.Size; i++)
            {
                Rectangle brect = CvInvoke.BoundingRectangle(contours[i]);

                double ar = brect.Width / brect.Height; // aspect ration (width > height)
                if (ar > 2 && brect.Width > 30 && brect.Height > 10 && brect.Height < 100)
                {
                    list.Add(brect);
                }
            }

            Image<Bgr, byte> imgOut = img.CopyBlank();
            foreach (var r in list)
            {
                CvInvoke.Rectangle(img, r, new MCvScalar(0, 0, 255), 2);
                CvInvoke.Rectangle(imgOut, r, new MCvScalar(0, 255, 255), -1); // -1 filled rectangle
            }

            imgOut._And(img); // Real Area will be copied to imgOut

            pictureBox1.Image = img.Bitmap;
            pictureBox2.Image = imgOut.Bitmap;
        }
    }
}
