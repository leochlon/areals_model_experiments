from setup_confirmation import ROOT,write


def main():
    """The six fresh-decoder models: Point-MAE or AREALS encoder behind a fresh query decoder, three seeds each."""
    tasks=[]
    for seed in range(3):
        for backbone in ['pointmae','areals']:
            tasks.append(dict(task=len(tasks),backbone=backbone,seed=seed,
                group='fresh_decoder_'+backbone,name=f'fresh_decoder_{backbone}_seed{seed}',
                checkpoint_source=str(ROOT/'results/architecture/reference'/f'seed{seed}/best.pt')))
    protocol=dict(comparisons=[['isotropic_reference','fresh_decoder_pointmae'],['isotropic_reference','fresh_decoder_areals']],
        statistics='All six contrasts: car-grouped 10000-draw Bonferroni99.166667% intervals.')
    write(ROOT/'decoder_controls_tasks.json',tasks);write(ROOT/'decoder_controls_protocol.json',protocol)
    print('decoder controls',len(tasks),'tasks',flush=True)


if __name__=='__main__':main()
