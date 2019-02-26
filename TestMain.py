import time
from sklearn.externals import joblib
from src.br.cefetrj.commom.media.Image import Image
from src.br.cefetrj.commom.media.DataPreparator import DataPreparator
from src.br.cefetrj.commom.Configuration import Configuration
from src.br.cefetrj.mosquitoes.VideoLoader import VideoLoader
from src.br.cefetrj.commom.utilities.Correlation import Correlation
from matplotlib import pyplot as plt
from skimage.transform import resize
from src.br.cefetrj.commom.utilities.GPS import GPSCoordinates
import gmplot
import urllib

start_time = time.time()

config = Configuration()

# Load test videos from dataset
videos_test = VideoLoader.load(video_root_path=config.getattribute('videos.test.rootPath'),
                                  video_extension=config.getattribute('videos.extension'),
                                  annotation_root_path=config.getattribute('annotations.test.rootPath'),
                                  annotation_extension=config.getattribute('annotations.extension'),
                                  standard_size=[config.getattribute('videoSettings.standardSize.height'),
                                                 config.getattribute('videoSettings.standardSize.width')],
                                  percentage_edge_remove=config.getattribute('videoSettings.percentageOfEdgeToRemove'))

# Load previously trained model
trainedModel_path = config.getattribute('dataset.rootPath')
rf = joblib.load(trainedModel_path + '/trainedRF.pkl')

# Get first video from training videos dataset
video = videos_test[0]

# Load GPS Table path and video frame length. Create gps object
gps_path = (config.getattribute('dataset.rootPath')+'/tabela.csv')
frame_length = video.num_frames
gps = GPSCoordinates()
gps.settable(gps_path,frame_length)

# Get only annotated frames
#annotation = video.getannotation().frameswithannotation

frames = list(range(540, 560))

# start object variable with empty
objects = {}

#for number_frame in [annotation[29], annotation[30], annotation[31], annotation[32]]:
for number_frame in frames:

    # Get first annotated frame
    image = video.getframe(number_frame)

    # Resize image for applying classifier

    image = resize(image, [int(image.shape[0]/3.5), int(image.shape[1]/3.5)], mode='reflect')

    # Split image into 20x20 squares for post processing
    images, positions = Image.split(image, [20, 20], 10, 10)

    # Get data for applying into trained classifier
    data = DataPreparator.getdatainimagelist(images, 'src.br.cefetrj.mosquitoes.FeatureExtractor.FeatureExtractor.fe1')

    # Applying classifier
    predictions = rf.predict(data)

    # Get objects localed
    matches = [positions[i] for i in range(0, len(predictions)) if predictions[i] == 1]

    # Apply correlation
    matches = Correlation.geometric(matches, config.getattribute('detection.distanceInFrame'))

    # Apply temporal correlation between frames
    objects = Correlation.temporalobjectcorrelation(objects,
                                                    matches,
                                                    number_frame,
                                                    config.getattribute('detection.distanceBetweenFrames')
                                                    )

# Remove low detected objects of dictionary

# objects = Correlation.temporalremovelowdetection(objects, config.getattribute('detection.minimumDetectionPerObject'))

objects = Correlation.temporalremovelowdetection(objects, 1)

# Get number of detected objects in frame
n_objects = len(objects.keys())

print("--- %s seconds ---" % (time.time() - start_time))

# Code to view result in image

for object_key1 in objects.keys():
    for object_key in objects[object_key1].keys():
        # Get first annotated frame
        image = video.getframe(int(object_key))

        #Get GPS location and plot map
        n_frame = int(object_key)
        lat, long = gps.getinterpolatetable(n_frame)
        gps.plotmap(lat, long, n_objects)

        # Resize image for applying classifier

        image = resize(image, [int(image.shape[0]/3.5), int(image.shape[1]/3.5)], mode='reflect')

        pstn = objects[object_key1][object_key]

        import matplotlib.patches as patches
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1)
        plt.imshow(image)

        rect = patches.Rectangle((pstn['x']['begin'], pstn['y']['begin']),
                                 (pstn['x']['end'] - pstn['x']['begin'] + 1),
                                 (pstn['y']['end'] - pstn['y']['begin'] + 1),
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()